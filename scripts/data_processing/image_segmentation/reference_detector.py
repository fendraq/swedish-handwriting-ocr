import cv2
import numpy as np
from typing import Dict, Tuple, List
import math
from PIL import Image

"""
Reference marker detection in scanned images.

Detecting reference markers (circles with cross) in the scanned document
to calculate transformation from PDF-coordinates to image-coordinates
"""

class ReferenceDetector:
    def __init__(self, marker_radius_mm=4):
        self.marker_radius_mm = marker_radius_mm

    def _detect_image_properties(self, image_path: str) -> dict:
        """
        Automatically detect image properties from scanned image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing width, height, actual_dpi, estimated_dpi, final_dpi
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Get DPI from metadata
                dpi_info = img.info.get('dpi')
                if dpi_info:
                    dpi_x, dpi_y = dpi_info
                    actual_dpi = int((dpi_x + dpi_y) / 2)  # Average
                else:
                    actual_dpi = None
                
                # Estimate DPI based on A4 size if no metadata DPI
                estimated_dpi = None
                if actual_dpi is None:
                    # A4 is 210x297 mm
                    # Assume image is A4 to estimate DPI
                    estimated_dpi_w = width / (210 / 25.4)
                    estimated_dpi_h = height / (297 / 25.4)
                    estimated_dpi = int((estimated_dpi_w + estimated_dpi_h) / 2)
                
                return {
                    'width': width,
                    'height': height,
                    'actual_dpi': actual_dpi,
                    'estimated_dpi': estimated_dpi,
                    'final_dpi': actual_dpi if actual_dpi else estimated_dpi
                }
                
        except Exception as e:
            print(f"Error reading image properties: {e}")
            return None

    def _calculate_marker_radius_px(self, marker_radius_mm: float, dpi: int) -> int:
        """
        Calculate marker radius in pixels based on DPI.
        
        Args:
            marker_radius_mm: Radius in millimeters
            dpi: Dots per inch
            
        Returns:
            Radius in pixels
        """
        marker_radius_inches = marker_radius_mm / 25.4
        marker_radius_px = int(marker_radius_inches * dpi)
        return marker_radius_px

    def _get_adaptive_hough_params(self, dpi: int, image_shape: Tuple[int, int, int]) -> dict:
        """
        Calculate adaptive HoughCircles parameters based on image properties.
        
        Args:
            dpi: Detected DPI
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Dict with HoughCircles parameters
        """
        h, w = image_shape[:2]
        marker_radius_px = self._calculate_marker_radius_px(self.marker_radius_mm, dpi)
        
        return {
            'dp': 1,
            'minDist': min(h, w) // 4,
            'param1': 50,
            'param2': 30,
            'minRadius': max(1, marker_radius_px - 15),
            'maxRadius': marker_radius_px + 15
        }

    def detect_circular_markers(self, image_path: str) -> Dict[str, Tuple[int, int]]:
        """
        Detect circular reference markers in image using dynamic analysis.

        Args:
            image_path: Path to input image file

        Returns:
            Dictionary mapping marker positions to coordinates
        """
        # Detect image properties
        props = self._detect_image_properties(image_path)
        if props is None:
            print("Failed to read image properties")
            return {}
        
        print(f"Image properties: {props['width']}x{props['height']}, DPI: {props['final_dpi']}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return {}

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Get adaptive parameters
        hough_params = self._get_adaptive_hough_params(props['final_dpi'], image.shape)
        print(f"Using adaptive parameters: minRadius={hough_params['minRadius']}, maxRadius={hough_params['maxRadius']}")

        # Use HoughCircles to find circular markers
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT,
            dp=hough_params['dp'],
            minDist=hough_params['minDist'],
            param1=hough_params['param1'],
            param2=hough_params['param2'],
            minRadius=hough_params['minRadius'],
            maxRadius=hough_params['maxRadius']
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')
            print(f"Detected {len(circles)} potential circular markers")

            # Filter circles that actually contain crosses
            valid_markers = self._filter_circles_with_crosses(gray, circles)

            # Match valid circles to expected positions
            return self._match_circles_to_positions(valid_markers, image.shape)
        
        print("No circular markers detected")
        return {}
    
    def _filter_circles_with_crosses(self, gray_image: np.ndarray, circles: np.ndarray) -> List[Tuple[int, int, int]]:
        """Filter circles that contain cross pattern"""
        valid_circles = []

        for (x, y, r) in circles:
            #Extract region around circle
            margin = 5
            x1, y1 = max(0, x - r - margin), max(0, y - r - margin)
            x2, y2 = min(gray_image.shape[1], x + r + margin), min(gray_image.shape[0], y + r + margin)

            roi = gray_image[y1:y2, x1:x2]

            # Check for cross pattern using line detection
            if self._has_cross_pattern(roi, r):
                valid_circles.append((x, y, r))
                print(f"Valid marker found at ({x}, {y}) with radius {r}")

        return valid_circles
    
    def _has_cross_pattern(self, roi: np.ndarray, expected_radius: int) -> bool:
        """Check if ROI contains a cross pattern - with lenient validation"""
        # Edge detection
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)

        # Use HoughLinesP to detect line segments
        lines = cv2.HoughLinesP(
            edges, 
            rho=1,
            theta=np.pi/180,
            threshold=max(5, expected_radius // 4),  # Lower threshold
            minLineLength=max(2, expected_radius // 3),  # Shorter minimum length
            maxLineGap=10  # Allow larger gaps
        )

        if lines is None:
            # If no lines detected, be more lenient and just check for sufficient edge content
            edge_pixels = cv2.countNonZero(edges)
            roi_area = roi.shape[0] * roi.shape[1]
            edge_ratio = edge_pixels / roi_area
            
            # If there's reasonable edge content, assume it might be a marker
            if edge_ratio > 0.05:  # 5% of pixels are edges
                print(f"    No lines detected but found {edge_ratio:.3f} edge ratio - accepting")
                return True
            return False
        
        # Check for roughly perpendicular lines
        horizontal_lines = []
        vertical_lines = []
        diagonal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

            # Horizontal-ish lines
            if abs(angle) < 25 or abs(angle) > 155:
                horizontal_lines.append(line)
            # Vertical-ish lines  
            elif 65 < abs(angle) < 115:
                vertical_lines.append(line)
            else:
                diagonal_lines.append(line)

        # Accept if we have both horizontal and vertical, OR reasonable line content
        has_cross = len(horizontal_lines) > 0 and len(vertical_lines) > 0
        has_lines = len(lines) >= 2  # At least some line content
        
        result = has_cross or has_lines
        print(f"    Cross validation: H={len(horizontal_lines)}, V={len(vertical_lines)}, D={len(diagonal_lines)}, Total={len(lines)} -> {'PASS' if result else 'FAIL'}")
        
        return result
    
    def _match_circles_to_positions(self, circles: List[Tuple[int, int, int]], image_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
        """Match detected circles to expected corner positions."""
        h, w = image_shape[:2]
        detected_markers = {}            

        print(f"Matching {len(circles)} circles to positions (image: {w}x{h})")
        
        for (x, y, r) in circles:
            print(f"  Circle at ({x}, {y}) radius {r}")
            
            # More lenient position matching - use thirds instead of strict boundaries
            position = None
            if x < w/2 and y < h/2:
                position = 'top_left'
            elif x > w/2 and y < h/2:
                position = 'top_right'
            elif x < w/2 and y > h/2:
                position = 'bottom_left'
            elif x > w/2 and y > h/2:
                position = 'bottom_right'
            
            if position:
                # If this position is already taken, keep the one closer to the actual corner
                if position in detected_markers:
                    # Calculate distances to ideal corner positions
                    corner_positions = {
                        'top_left': (0, 0),
                        'top_right': (w, 0),
                        'bottom_left': (0, h),
                        'bottom_right': (w, h)
                    }
                    
                    ideal_x, ideal_y = corner_positions[position]
                    
                    # Current marker distance
                    current_x, current_y = detected_markers[position]
                    current_dist = ((current_x - ideal_x)**2 + (current_y - ideal_y)**2)**0.5
                    
                    # New marker distance
                    new_dist = ((x - ideal_x)**2 + (y - ideal_y)**2)**0.5
                    
                    if new_dist < current_dist:
                        print(f"    Replacing {position}: ({current_x}, {current_y}) with ({x}, {y}) - closer to corner")
                        detected_markers[position] = (x, y)
                    else:
                        print(f"    Keeping existing {position}: ({current_x}, {current_y}) - closer than ({x}, {y})")
                else:
                    print(f"    Assigned to {position}")
                    detected_markers[position] = (x, y)
            else:
                print(f"    No position match for ({x}, {y})")

        print(f"Final matched markers: {list(detected_markers.keys())}")
        return detected_markers
    
    def calculate_transformation(self, detected_markers: Dict[str, Tuple[int, int]], pdf_markers:Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        Calculate transformation matrix from PDF coordinates to image coordinates.

        Args:
            detected_markers: Detected marker positions in image
            pdf_markers: Reference marker positions from PDF metadata

        Returns:
            3x3 perspective transformation matrix
        """

        if len(detected_markers) < 2:
            raise ValueError(f"Need at least 2 reference markers for transformation, got {len(detected_markers)}")

        pdf_points = []
        img_points = []

        # Use consistent ordering for transformation
        marker_order = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

        for marker_id in marker_order:
            if marker_id in detected_markers and marker_id in pdf_markers:
                pdf_points.append(pdf_markers[marker_id])
                img_points.append(detected_markers[marker_id])

        if len(pdf_points) < 2:
            raise ValueError("Need at least 2 matching markers for transformation")
            
        pdf_pts = np.float32(pdf_points)
        img_pts = np.float32(img_points)

        if len(pdf_points) == 4:
            # Full perspective transformation
            transformation_matrix = cv2.getPerspectiveTransform(pdf_pts, img_pts)
        elif len(pdf_points) == 3:
            # Affine transformation for 3 points
            transformation_matrix = cv2.getAffineTransform(pdf_pts[:3], img_pts[:3])
            # Convert to 3x3 matrix
            affine_3x3 = np.zeros((3, 3))
            affine_3x3[:2, :] = transformation_matrix
            affine_3x3[2, 2] = 1
            transformation_matrix = affine_3x3
        else:
            # For 2 points, use similarity transformation (scale, rotation, translation)
            print("Using 2-point similarity transformation")
            
            # Calculate scale and rotation from the two points
            pdf_p1, pdf_p2 = pdf_pts[0], pdf_pts[1]
            img_p1, img_p2 = img_pts[0], img_pts[1]
            
            # Calculate vectors
            pdf_vec = pdf_p2 - pdf_p1
            img_vec = img_p2 - img_p1
            
            # Calculate scale
            pdf_dist = np.linalg.norm(pdf_vec)
            img_dist = np.linalg.norm(img_vec)
            scale = img_dist / pdf_dist if pdf_dist > 0 else 1.0
            
            # Calculate rotation
            pdf_angle = math.atan2(pdf_vec[1], pdf_vec[0])
            img_angle = math.atan2(img_vec[1], img_vec[0])
            rotation = img_angle - pdf_angle
            
            # Create transformation matrix
            cos_r = math.cos(rotation) * scale
            sin_r = math.sin(rotation) * scale
            
            # Translation to align first point
            tx = img_p1[0] - (cos_r * pdf_p1[0] - sin_r * pdf_p1[1])
            ty = img_p1[1] - (sin_r * pdf_p1[0] + cos_r * pdf_p1[1])
            
            transformation_matrix = np.array([
                [cos_r, -sin_r, tx],
                [sin_r,  cos_r, ty],
                [0,      0,     1]
            ], dtype=np.float32)

        print(f"Transformation matrix calculated using {len(pdf_points)} points")
        return transformation_matrix
    
    def visualize_detected_markers(self, image: np.ndarray, detected_markers: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """
        Draw detected markers on image for visualization

        Args: 
            image: Input image
            detected_markers: Detected marker positions

        Returns:
            Image with markes highlighted
        """

        vis_image = image.copy()

        colors = {
            'top_left': (0, 255, 0),
            'top_right': (255, 0, 0),
            'bottom_left': (0, 255, 255),
            'bottom_right': (255, 0, 255)
        }

        for marker_id, (x, y) in detected_markers.items():
            color = colors.get(marker_id, (255, 255, 255))

            # Draw circle
            cv2.circle(vis_image, (x, y), 3, color, -1)

            # Draw label
            cv2.putText(vis_image, marker_id[:2].upper(), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return vis_image