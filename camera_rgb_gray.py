import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time

class SimpleAutoSegmentation:
    def __init__(self, checkpoint, model_cfg, width=640, height=480, fps=30):
        """初始化 RealSense 和 SAM2"""
        
        # 初始化 SAM2
        print("載入 SAM2 模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {self.device}")
        
        sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        # 初始化 RealSense
        print("初始化 RealSense...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 檢查可用的設備
        self.check_realsense_devices()
        
        # 啟用彩色和深度流
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # 初始化對齊對象（將深度對齊到彩色）
        self.align = rs.align(rs.stream.color)
        
        # 狀態變數
        self.current_frame = None
        self.current_depth = None
        self.mask = None
        self.frame_count = 0
        
        # 深度可視化相關
        self.combined_window_created = False
        
        print("初始化完成！")
    
    def check_realsense_devices(self):
        """檢查 RealSense 設備連接狀態"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            print(f"找到 {len(devices)} 個 RealSense 設備")
            
            if len(devices) == 0:
                print("錯誤: 沒有找到 RealSense 設備!")
                print("請檢查:")
                print("1. 設備是否正確連接到 USB 3.0 端口")
                print("2. USB 線是否正常工作")
                print("3. 設備驅動是否正確安裝")
                raise RuntimeError("沒有找到 RealSense 設備")
            
            for i, device in enumerate(devices):
                print(f"設備 {i}: {device.get_info(rs.camera_info.name)}")
                print(f"  序號: {device.get_info(rs.camera_info.serial_number)}")
                print(f"  固件版本: {device.get_info(rs.camera_info.firmware_version)}")
                
                # 檢查設備是否支援所需的感測器
                sensors = device.query_sensors()
                has_color = False
                has_depth = False
                
                for sensor in sensors:
                    if sensor.is_color_sensor():
                        has_color = True
                        print(f"  ✓ 支援彩色感測器")
                    if sensor.is_depth_sensor():
                        has_depth = True
                        print(f"  ✓ 支援深度感測器")
                
                if not has_color or not has_depth:
                    print(f"  ⚠ 警告: 設備缺少必要的感測器")
                    
        except Exception as e:
            print(f"檢查設備時發生錯誤: {e}")
            raise
    
    def initialize_pipeline_with_retry(self, max_retries=3):
        """嘗試啟動 pipeline，包含重試機制"""
        for attempt in range(max_retries):
            try:
                print(f"嘗試啟動 RealSense pipeline (第 {attempt + 1} 次)...")
                
                # 如果不是第一次嘗試，先重置 pipeline
                if attempt > 0:
                    try:
                        self.pipeline.stop()
                    except:
                        pass
                    
                    # 等待一下再重試
                    time.sleep(2)
                    
                    # 重新創建 pipeline 和 config
                    self.pipeline = rs.pipeline()
                    self.config = rs.config()
                    self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                
                # 嘗試啟動
                profile = self.pipeline.start(self.config)
                
                # 檢查是否成功獲取流
                print("測試是否能獲取幀...")
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    print("✓ RealSense pipeline 啟動成功!")
                    return True
                else:
                    print("✗ 無法獲取有效的彩色或深度幀")
                    self.pipeline.stop()
                    
            except Exception as e:
                print(f"第 {attempt + 1} 次嘗試失敗: {e}")
                
                if "failed to set power state" in str(e):
                    print("電源狀態設置失敗，可能的解決方案:")
                    print("1. 拔掉 USB 線，等待 5 秒後重新插入")
                    print("2. 嘗試不同的 USB 3.0 端口")
                    print("3. 重啟相機設備管理程序")
                    print("4. 檢查是否有其他程序正在使用相機")
                
                if attempt < max_retries - 1:
                    print(f"等待 3 秒後重試...")
                    time.sleep(3)
        
        print("所有嘗試都失敗了")
        return False
    
    def auto_segment(self, image):
        """自動分割圖像"""
        try:
            # 設置圖像到 SAM2
            self.predictor.set_image(image)
            
            # 在圖像中央創建一個點作為提示
            h, w = image.shape[:2]
            center_point = np.array([[w//2, h//2]])
            center_label = np.array([1])
            
            # 執行分割
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=center_point,
                    point_labels=center_label,
                    multimask_output=True,
                )
            
            if len(masks) > 0:
                # 選擇最佳遮罩
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                
                print(f"分割完成，得分: {scores[best_idx]:.3f}")
                print(f"遮罩類型: {mask.dtype}, 形狀: {mask.shape}")
                
                return mask
                
        except Exception as e:
            print(f"分割錯誤: {e}")
            import traceback
            traceback.print_exc()
            
        return None
    
    def extract_depth_from_mask(self, depth_image, mask):
        """從遮罩區域提取深度信息（優化版本）"""
        if mask is None or depth_image is None:
            return None, None, None
        
        try:
            # 確保遮罩是布林類型
            if mask.dtype != bool:
                mask = mask > 0.5
            
            # 直接將非分割區域的深度設為0，節省計算資源
            masked_depth = np.zeros_like(depth_image, dtype=np.uint16)
            masked_depth[mask] = depth_image[mask]
            
            # 獲取有效的深度值（非零值）
            valid_depths = masked_depth[masked_depth > 0]
            
            if len(valid_depths) == 0:
                return None, None, None
            
            # 計算統計信息
            mean_depth = np.mean(valid_depths)
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            
            print(f"深度統計 - 平均: {mean_depth:.1f}mm, 最近: {min_depth}mm, 最遠: {max_depth}mm")
            
            return masked_depth, valid_depths, {
                'mean': mean_depth,
                'min': min_depth,
                'max': max_depth,
                'count': len(valid_depths)
            }
            
        except Exception as e:
            print(f"提取深度錯誤: {e}")
            return None, None, None
    
    def draw_overlay_with_depth_info(self, image, mask, depth_image, alpha=0.5):
        """繪製分割遮罩疊加並顯示深度信息"""
        if mask is None:
            return image
        
        try:
            # 確保遮罩是布林類型
            if mask.dtype != bool:
                mask = mask > 0.5
            
            result = image.copy()
            
            # 創建彩色遮罩
            colored_mask = np.zeros_like(image)
            colored_mask[mask] = [0, 255, 0]  # 綠色
            
            # 混合原圖和遮罩
            result = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
            
            # 繪製邊界
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            
            # 添加深度信息顯示
            if depth_image is not None:
                # 計算物體的深度統計
                object_depths = depth_image[mask]
                valid_depths = object_depths[object_depths > 0]
                
                if len(valid_depths) > 0:
                    mean_depth = np.mean(valid_depths)
                    min_depth = np.min(valid_depths)
                    max_depth = np.max(valid_depths)
                    
                    # 在物體中心顯示平均深度
                    moments = cv2.moments(mask_uint8)
                    if moments["m00"] != 0:
                        center_x = int(moments["m10"] / moments["m00"])
                        center_y = int(moments["m01"] / moments["m00"])
                        
                        # 顯示中心點深度
                        center_depth = depth_image[center_y, center_x] if depth_image[center_y, center_x] > 0 else mean_depth
                        
                        # 繪製中心點
                        cv2.circle(result, (center_x, center_y), 5, (255, 255, 0), -1)
                        
                        # 顯示深度數值
                        depth_text = f"{center_depth:.0f}mm"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                        
                        # 計算文字大小以添加背景
                        text_size = cv2.getTextSize(depth_text, font, font_scale, thickness)[0]
                        
                        # 添加文字背景
                        text_x = center_x - text_size[0] // 2
                        text_y = center_y - 20
                        
                        # 確保文字不會超出圖像邊界
                        text_x = max(0, min(text_x, image.shape[1] - text_size[0]))
                        text_y = max(text_size[1], min(text_y, image.shape[0] - 10))
                        
                        cv2.rectangle(result, 
                                    (text_x - 5, text_y - text_size[1] - 5),
                                    (text_x + text_size[0] + 5, text_y + 5),
                                    (0, 0, 0), -1)
                        
                        cv2.putText(result, depth_text, (text_x, text_y), 
                                  font, font_scale, (255, 255, 0), thickness)
                    
                    # 在圖像頂部顯示詳細深度統計
                    stats_y = 30
                    font_scale_small = 0.6
                    thickness_small = 1
                    
                    depth_stats = [
                        f"Object Depth Stats:",
                        f"Mean: {mean_depth:.1f}mm",
                        f"Range: {min_depth:.0f}-{max_depth:.0f}mm",
                        f"Distance: {mean_depth/10:.1f}cm"
                    ]
                    
                    for i, stat in enumerate(depth_stats):
                        # 添加文字背景
                        text_size_small = cv2.getTextSize(stat, font, font_scale_small, thickness_small)[0]
                        cv2.rectangle(result, 
                                    (10, stats_y + i*25 - text_size_small[1] - 3),
                                    (10 + text_size_small[0] + 6, stats_y + i*25 + 3),
                                    (0, 0, 0), -1)
                        
                        color = (0, 255, 255) if i == 0 else (255, 255, 255)
                        cv2.putText(result, stat, (13, stats_y + i*25), 
                                  font, font_scale_small, color, thickness_small)
            
            return result
            
        except Exception as e:
            print(f"繪製覆蓋層錯誤: {e}")
            print(f"遮罩類型: {mask.dtype}, 形狀: {mask.shape}")
            return image
    
    def create_detailed_depth_visualization(self, rgb_image, mask, depth_image):
        """創建詳細的深度可視化，突出顯示物體深度"""
        try:
            if mask is None or depth_image is None:
                return rgb_image
            
            h, w = rgb_image.shape[:2]
            
            # 確保遮罩是布林類型
            if mask.dtype != bool:
                mask_bool = mask > 0.5
            else:
                mask_bool = mask
            
            # 提取物體深度
            object_depths = depth_image[mask_bool]
            valid_depths = object_depths[object_depths > 0]
            
            if len(valid_depths) == 0:
                return rgb_image
            
            # 計算深度統計
            mean_depth = np.mean(valid_depths)
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            std_depth = np.std(valid_depths)
            
            # 創建深度熱力圖
            depth_heatmap = np.zeros_like(depth_image, dtype=np.float32)
            depth_heatmap[mask_bool] = depth_image[mask_bool].astype(np.float32)
            
            # 正規化到0-1範圍（只針對物體區域）
            if max_depth > min_depth:
                normalized_heatmap = np.zeros_like(depth_heatmap)
                object_mask = depth_heatmap > 0
                normalized_heatmap[object_mask] = (depth_heatmap[object_mask] - min_depth) / (max_depth - min_depth)
            else:
                normalized_heatmap = depth_heatmap
            
            # 轉換為8位並應用顏色映射
            heatmap_8bit = (normalized_heatmap * 255).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
            
            # 非物體區域保持黑色
            colored_heatmap[~mask_bool] = [0, 0, 0]
            
            # 混合原圖和熱力圖
            alpha = 0.6
            result = cv2.addWeighted(rgb_image, 1-alpha, colored_heatmap, alpha, 0)
            
            # 添加物體輪廓
            mask_uint8 = mask_bool.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (255, 255, 255), 3)
            
            # 添加深度資訊面板
            panel_height = 120
            panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            color = (255, 255, 255)
            
            # 深度統計資訊
            info_lines = [
                f"=== OBJECT DEPTH ANALYSIS ===",
                f"Mean Depth: {mean_depth:.1f} mm ({mean_depth/10:.1f} cm)",
                f"Depth Range: {min_depth:.0f} - {max_depth:.0f} mm",
                f"Variation: ±{std_depth:.1f} mm",
                f"Object Size: {np.sum(mask_bool)} pixels"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 20 + i * 20
                if i == 0:  # 標題用不同顏色
                    cv2.putText(panel, line, (10, y_pos), font, font_scale, (0, 255, 255), thickness)
                else:
                    cv2.putText(panel, line, (10, y_pos), font, font_scale, color, thickness)
            
            # 添加顏色條說明
            colorbar_x = w - 200
            colorbar_y = 20
            colorbar_width = 20
            colorbar_height = 80
            
            # 創建顏色條
            for i in range(colorbar_height):
                color_val = int(255 * i / colorbar_height)
                color_bgr = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                cv2.rectangle(panel, 
                            (colorbar_x, colorbar_y + colorbar_height - i), 
                            (colorbar_x + colorbar_width, colorbar_y + colorbar_height - i + 1),
                            color_bgr.tolist(), -1)
            
            # 顏色條標籤
            cv2.putText(panel, f"{max_depth:.0f}mm", (colorbar_x + 25, colorbar_y + 15), 
                       font, 0.4, color, 1)
            cv2.putText(panel, f"{min_depth:.0f}mm", (colorbar_x + 25, colorbar_y + colorbar_height), 
                       font, 0.4, color, 1)
            cv2.putText(panel, "Near", (colorbar_x + 25, colorbar_y + colorbar_height + 15), 
                       font, 0.4, (0, 0, 255), 1)
            cv2.putText(panel, "Far", (colorbar_x + 25, colorbar_y), 
                       font, 0.4, (255, 0, 0), 1)
            
            # 合併圖像和資訊面板
            combined = np.vstack([result, panel])
            
            return combined
            
        except Exception as e:
            print(f"創建詳細深度可視化錯誤: {e}")
            return rgb_image
    def create_combined_visualization(self, rgb_image, mask, depth_image):
        """創建四合一可視化：RGB原圖 + 分割遮罩 + 深度圖 + 深度分析"""
        try:
            h, w = rgb_image.shape[:2]
            
            # 1. RGB原圖
            rgb_display = rgb_image.copy()
            
            # 2. 分割遮罩圖（非目標區域變黑）
            if mask is not None:
                # 確保遮罩是布林類型
                if mask.dtype != bool:
                    mask_bool = mask > 0.5
                else:
                    mask_bool = mask
                
                # 創建遮罩圖像：目標區域保持原色，其他區域變黑
                masked_image = np.zeros_like(rgb_image)
                masked_image[mask_bool] = rgb_image[mask_bool]
                
                # 添加深度數值到遮罩圖像
                if depth_image is not None:
                    object_depths = depth_image[mask_bool]
                    valid_depths = object_depths[object_depths > 0]
                    if len(valid_depths) > 0:
                        mean_depth = np.mean(valid_depths)
                        # 在遮罩圖像上顯示平均深度
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        depth_text = f"Avg: {mean_depth:.0f}mm"
                        cv2.putText(masked_image, depth_text, (10, h-20), 
                                  font, 0.7, (0, 255, 255), 2)
            else:
                masked_image = np.zeros_like(rgb_image)
            
            # 3. 深度圖（只顯示分割區域的深度，優化版本）
            if depth_image is not None and mask is not None:
                # 直接創建只有分割區域的深度圖，節省計算資源
                masked_depth = np.zeros_like(depth_image, dtype=np.uint16)
                masked_depth[mask_bool] = depth_image[mask_bool]
                
                # 獲取有效深度範圍進行正規化
                valid_depths = masked_depth[masked_depth > 0]
                if len(valid_depths) > 0:
                    min_depth = np.min(valid_depths)
                    max_depth = np.max(valid_depths)
                    
                    # 正規化深度值到0-255範圍
                    normalized_depth = np.zeros_like(depth_image, dtype=np.uint8)
                    if max_depth > min_depth:
                        # 只對有效深度進行正規化，其他區域自動為0
                        valid_mask = masked_depth > 0
                        normalized_depth[valid_mask] = ((masked_depth[valid_mask] - min_depth) / 
                                                      (max_depth - min_depth) * 255).astype(np.uint8)
                    
                    # 應用色彩映射
                    depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
                    # 非目標區域自動保持黑色（因為normalized_depth中已經是0）
                    depth_colormap[~mask_bool] = [0, 0, 0]
                    
                    # 添加深度範圍資訊
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    range_text = f"{min_depth:.0f}-{max_depth:.0f}mm"
                    cv2.putText(depth_colormap, range_text, (10, h-20), 
                              font, 0.6, (255, 255, 255), 2)
                else:
                    depth_colormap = np.zeros_like(rgb_image)
            else:
                depth_colormap = np.zeros_like(rgb_image)
            
            # 4. 詳細深度分析圖
            detail_depth = self.create_detailed_depth_visualization(rgb_image, mask, depth_image)
            
            # 調整圖像大小以便並排顯示（2x2格局）
            display_height = 300
            display_width = int(w * display_height / h)
            
            rgb_resized = cv2.resize(rgb_display, (display_width, display_height))
            mask_resized = cv2.resize(masked_image, (display_width, display_height))
            depth_resized = cv2.resize(depth_colormap, (display_width, display_height))
            detail_resized = cv2.resize(detail_depth, (display_width, display_height))
            
            # 添加標題
            title_height = 30
            title_color = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # 為每個圖像添加標題區域
            def add_title(img, title):
                titled_img = np.zeros((display_height + title_height, display_width, 3), dtype=np.uint8)
                cv2.putText(titled_img, title, (10, 20), font, font_scale, title_color, thickness)
                titled_img[title_height:, :] = img
                return titled_img
            
            rgb_with_title = add_title(rgb_resized, "RGB Original")
            mask_with_title = add_title(mask_resized, "Segmentation + Depth")
            depth_with_title = add_title(depth_resized, "Depth Heatmap")
            detail_with_title = add_title(detail_resized, "Depth Analysis")
            
            # 2x2排列
            top_row = np.hstack([rgb_with_title, mask_with_title])
            bottom_row = np.hstack([depth_with_title, detail_with_title])
            combined = np.vstack([top_row, bottom_row])
            
            # 添加分隔線
            line_color = (100, 100, 100)
            line_thickness = 2
            total_height, total_width = combined.shape[:2]
            
            # 垂直分隔線
            cv2.line(combined, (display_width, 0), (display_width, total_height), line_color, line_thickness)
            # 水平分隔線
            cv2.line(combined, (0, display_height + title_height), (total_width, display_height + title_height), line_color, line_thickness)
            
            return combined
            
        except Exception as e:
            print(f"創建合併可視化錯誤: {e}")
            # 返回原始圖像作為備選
            return rgb_image

    def create_depth_matrix_visualization(self, depth_image, mask, region_size=(20, 20)):
        """創建深度數值矩陣可視化（優化版本）"""
        try:
            if mask is None or depth_image is None:
                return np.zeros((300, 600, 3), dtype=np.uint8)
            
            # 確保遮罩是布林類型
            if mask.dtype != bool:
                mask_bool = mask > 0.5
            else:
                mask_bool = mask
            
            # 找到遮罩的邊界框
            rows = np.any(mask_bool, axis=1)
            cols = np.any(mask_bool, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return np.zeros((300, 600, 3), dtype=np.uint8)
            
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # 計算中心區域
            center_r = (rmin + rmax) // 2
            center_c = (cmin + cmax) // 2
            
            # 定義顯示區域大小
            half_size_r, half_size_c = region_size[0] // 2, region_size[1] // 2
            
            # 確保不超出圖像邊界
            start_r = max(0, center_r - half_size_r)
            end_r = min(depth_image.shape[0], center_r + half_size_r)
            start_c = max(0, center_c - half_size_c)
            end_c = min(depth_image.shape[1], center_c + half_size_c)
            
            # 提取深度矩陣（優化：直接處理目標區域）
            depth_matrix = depth_image[start_r:end_r, start_c:end_c]
            mask_matrix = mask_bool[start_r:end_r, start_c:end_c]
            
            # 只保留分割區域的深度值，其他設為0
            display_depth_matrix = np.zeros_like(depth_matrix)
            display_depth_matrix[mask_matrix] = depth_matrix[mask_matrix]
            
            # 創建顯示圖像
            img_height = 600
            img_width = 800
            matrix_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 30  # 深色背景
            
            # 標題
            font = cv2.FONT_HERSHEY_SIMPLEX
            title_font_scale = 0.8
            value_font_scale = 0.4
            color = (255, 255, 255)
            
            # 添加標題
            title = f"Depth Matrix ({display_depth_matrix.shape[0]}x{display_depth_matrix.shape[1]}) - Center Region"
            cv2.putText(matrix_img, title, (10, 30), font, title_font_scale, color, 2)
            
            # 計算每個格子的大小
            matrix_h, matrix_w = display_depth_matrix.shape
            if matrix_h > 0 and matrix_w > 0:
                cell_width = min(30, (img_width - 40) // matrix_w)
                cell_height = min(25, (img_height - 100) // matrix_h)
                
                start_x = (img_width - matrix_w * cell_width) // 2
                start_y = 60
                
                # 繪製矩陣
                for i in range(matrix_h):
                    for j in range(matrix_w):
                        x = start_x + j * cell_width
                        y = start_y + i * cell_height
                        
                        depth_value = display_depth_matrix[i, j]
                        is_in_mask = mask_matrix[i, j]
                        
                        # 根據是否在遮罩內設置顏色
                        if is_in_mask and depth_value > 0:
                            # 在分割區域內且有有效深度值
                            bg_color = (0, 100, 0)  # 深綠色背景
                            text_color = (255, 255, 255)  # 白色文字
                            text = f"{depth_value}"
                        else:
                            # 非分割區域或無效深度值（優化後統一處理為0）
                            bg_color = (0, 0, 50)  # 深藍色背景
                            text_color = (100, 100, 100)  # 暗灰色文字
                            text = "0"
                        
                        # 繪製格子背景
                        cv2.rectangle(matrix_img, (x, y), (x + cell_width - 1, y + cell_height - 1), 
                                    bg_color, -1)
                        cv2.rectangle(matrix_img, (x, y), (x + cell_width - 1, y + cell_height - 1), 
                                    (100, 100, 100), 1)
                        
                        # 繪製數值
                        text_size = cv2.getTextSize(text, font, value_font_scale, 1)[0]
                        text_x = x + (cell_width - text_size[0]) // 2
                        text_y = y + (cell_height + text_size[1]) // 2
                        cv2.putText(matrix_img, text, (text_x, text_y), font, 
                                  value_font_scale, text_color, 1)
                
                # 添加圖例
                legend_y = start_y + matrix_h * cell_height + 30
                cv2.putText(matrix_img, "Legend:", (10, legend_y), font, 0.6, color, 1)
                
                # 綠色圖例
                cv2.rectangle(matrix_img, (10, legend_y + 10), (30, legend_y + 25), (0, 100, 0), -1)
                cv2.putText(matrix_img, "Target object depth (mm)", (40, legend_y + 22), 
                          font, 0.5, color, 1)
                
                # 藍色圖例
                cv2.rectangle(matrix_img, (10, legend_y + 35), (30, legend_y + 50), (0, 0, 50), -1)
                cv2.putText(matrix_img, "Non-target area (depth = 0)", (40, legend_y + 47), 
                          font, 0.5, color, 1)
                
                # 添加統計信息
                valid_depths = display_depth_matrix[display_depth_matrix > 0]
                if len(valid_depths) > 0:
                    stats_y = legend_y + 75
                    stats_text = [
                        f"Valid pixels: {len(valid_depths)}",
                        f"Mean depth: {np.mean(valid_depths):.1f} mm",
                        f"Min depth: {np.min(valid_depths):.1f} mm",
                        f"Max depth: {np.max(valid_depths):.1f} mm",
                        f"Std dev: {np.std(valid_depths):.1f} mm"
                    ]
                    
                    for i, text in enumerate(stats_text):
                        cv2.putText(matrix_img, text, (10, stats_y + i * 20), 
                                  font, 0.5, color, 1)
            
            return matrix_img
            
        except Exception as e:
            print(f"創建深度矩陣可視化錯誤: {e}")
            error_img = np.ones((300, 600, 3), dtype=np.uint8) * 30
            cv2.putText(error_img, "Error creating depth matrix", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return error_img
    
    def save_depth_matrix_to_file(self, depth_image, mask, filename="depth_matrix.txt"):
        """將深度矩陣保存到文件（優化版本）"""
        try:
            if mask is None or depth_image is None:
                return
            
            # 確保遮罩是布林類型
            if mask.dtype != bool:
                mask_bool = mask > 0.5
            else:
                mask_bool = mask
            
            # 直接創建優化的深度矩陣：非分割區域為0
            masked_depth = np.zeros_like(depth_image, dtype=np.uint16)
            masked_depth[mask_bool] = depth_image[mask_bool]
            
            # 找到邊界框
            rows = np.any(mask_bool, axis=1)
            cols = np.any(mask_bool, axis=0)
            
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                # 裁剪到邊界框
                cropped_depth = masked_depth[rmin:rmax+1, cmin:cmax+1]
                cropped_mask = mask_bool[rmin:rmax+1, cmin:cmax+1]
                
                # 保存到文件
                with open(filename, 'w') as f:
                    f.write(f"# Optimized Depth Matrix - Shape: {cropped_depth.shape}\n")
                    f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Values in mm (0 = non-target area)\n")
                    f.write(f"# Optimization: Non-segmentation areas set to 0 for resource saving\n\n")
                    
                    for i in range(cropped_depth.shape[0]):
                        row_values = []
                        for j in range(cropped_depth.shape[1]):
                            depth_val = cropped_depth[i, j]
                            # 優化：統一格式，非目標區域直接顯示0
                            row_values.append(f"{depth_val:4d}")
                        f.write(" ".join(row_values) + "\n")
                
                print(f"優化的深度矩陣已保存到: {filename}")
                
        except Exception as e:
            print(f"保存深度矩陣錯誤: {e}")
    
    def run(self):
        """主循環"""
        try:
            # 使用重試機制啟動 pipeline
            if not self.initialize_pipeline_with_retry():
                print("無法啟動 RealSense pipeline，程式退出")
                return
            
            # 創建合併可視化窗口
            cv2.namedWindow('Combined View', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Depth Matrix', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Auto Segmentation', cv2.WINDOW_AUTOSIZE)
            self.combined_window_created = True
            
            print("OpenCV 視窗創建成功!")
            
            print("\n自動分割啟動!")
            print("按 ESC 退出")
            print("視窗說明:")
            print("- Auto Segmentation: 即時深度分析 + 物體中心深度顯示")
            print("- Combined View: 2x2視圖 (RGB | 分割+深度 | 深度熱力圖 | 詳細分析)")
            print("- Depth Matrix: 分割區域的深度數值矩陣")
            print("按鍵說明:")
            print("- ESC: 退出程式")
            print("- S: 保存當前深度矩陣到文件")
            print("- 深度資訊特色:")
            print("  * 物體中心顯示即時深度 (mm)")
            print("  * 平均深度、範圍、變異統計")
            print("  * 深度熱力圖顯示距離分佈")
            print("  * 顏色編碼: 藍色=近距離, 紅色=遠距離")
            
            last_segment_time = 0
            segment_interval = 0.5  # 每0.5秒分割一次
            
            frame_received = False
            
            while True:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    
                    # 對齊深度到彩色
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        print("警告: 無法獲取彩色或深度幀")
                        continue
                    
                    if not frame_received:
                        print("成功接收到第一幀影像!")
                        frame_received = True
                
                    # 獲取圖像和深度數據
                    self.current_frame = np.asanyarray(color_frame.get_data())
                    self.current_depth = np.asanyarray(depth_frame.get_data())
                    display_frame = self.current_frame.copy()
                    
                    # 每隔一段時間執行分割
                    current_time = time.time()
                    if current_time - last_segment_time > segment_interval:
                        print("執行分割...")
                        self.mask = self.auto_segment(self.current_frame)
                        
                        if self.mask is not None:
                            # 提取深度信息（使用優化版本）
                            masked_depth, valid_depths, depth_stats = self.extract_depth_from_mask(
                                self.current_depth, self.mask
                            )
                            
                            # 創建合併可視化
                            if self.combined_window_created:
                                combined_view = self.create_combined_visualization(
                                    self.current_frame, self.mask, self.current_depth
                                )
                                cv2.imshow('Combined View', combined_view)
                                
                                # 創建深度矩陣可視化
                                depth_matrix_view = self.create_depth_matrix_visualization(
                                    self.current_depth, self.mask, region_size=(15, 20)
                                )
                                cv2.imshow('Depth Matrix', depth_matrix_view)
                        
                        last_segment_time = current_time
                    
                    # 繪製分割結果並顯示深度資訊
                    if self.mask is not None:
                        display_frame = self.draw_overlay_with_depth_info(display_frame, self.mask, self.current_depth)
                    
                    # 顯示狀態
                    cv2.putText(display_frame, f"Frame: {self.frame_count}", 
                               (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 顯示優化狀態
                    cv2.putText(display_frame, "Real-time Depth Analysis", 
                               (10, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow('Auto Segmentation', display_frame)
                    
                    self.frame_count += 1
                    
                    # 檢查退出
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord('s') or key == ord('S'):  # S鍵保存深度矩陣
                        if self.mask is not None and self.current_depth is not None:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"optimized_depth_matrix_{timestamp}.txt"
                            self.save_depth_matrix_to_file(self.current_depth, self.mask, filename)
                        
                except RuntimeError as re:
                    if "timeout" in str(re).lower():
                        print("幀超時，繼續嘗試...")
                        continue
                    else:
                        print(f"RealSense 運行時錯誤: {re}")
                        break
                except Exception as frame_error:
                    print(f"處理幀時發生錯誤: {frame_error}")
                    continue
                    
        except Exception as e:
            print(f"執行錯誤: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("正在關閉程式...")
            
            try:
                self.pipeline.stop()
            except:
                pass
            cv2.destroyAllWindows()
            print("程式結束")


def main():
    # 設置你的模型路徑
    checkpoint = "sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    try:
        print("=== 優化版 RealSense SAM2 自動分割系統 ===")
        print("優化特性:")
        print("1. 非分割區域深度直接設為0，節省計算資源")
        print("2. 移除重複程式碼，提升可讀性")
        print("3. 優化深度處理流程，提升效能")
        print("4. 改善記憶體使用效率")
        print()
        
        system = SimpleAutoSegmentation(
            checkpoint=checkpoint,
            model_cfg=model_cfg,
            width=640,
            height=480,
            fps=30
        )
        system.run()
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()