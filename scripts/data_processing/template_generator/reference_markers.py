"""
Reference marker generation for coordinate correction.

Creates reference markers (circles with cross) in the corners of the PDF to 
make precise coordinate transformation in image_segmentation.
"""

from reportlab.lib.units import mm

class ReferenceMarkerGenerator:
    def __init__(self, page_width, page_height, margin):
        self.page_width = page_width
        self.page_height = page_height
        self.margin = margin

        # Define marker positions
        offset = 13*mm
        self.markers = {
            'top_left': (offset, page_height - offset),
            'top_right': (page_width - offset, page_height - offset),
            'bottom_left': (offset, offset),
            'bottom_right': (page_width - offset, offset)
        }

        # Marker properties
        self.circle_radius = 4*mm
        self.cross_size = 2*mm

    def draw_markers(self, canvas_obj):
        """Draw reference markers on PDF canvas."""
        canvas_obj.setLineWidth(2)

        for marker_id, (x, y) in self.markers.items():
            canvas_obj.circle(x, y, self.circle_radius, stroke=1, fill=0)
            canvas_obj.line(x - self.cross_size, y, x + self.cross_size, y)
            canvas_obj.line(x, y - self.cross_size, x, y + self.cross_size)

            canvas_obj.setFont('Helvetica', 8)
            text_offset = self.circle_radius + 2*mm

            if 'left' in marker_id:
                canvas_obj.drawString(x + text_offset, y, marker_id[:2].upper())
            else:
                canvas_obj.drawRightString(x - text_offset, y, marker_id[:2].upper())

        canvas_obj.setLineWidth(1)

    def get_marker_metadata(self):
        """Return marker coordinates and properties for metadata."""
        return {
            'reference_markers': self.markers,
            'marker_type': 'circle_with_cross',
            'circle_radius_mm': self.circle_radius / mm,
            'cross_size_mm': self.cross_size / mm,
            'description': 'Four corner reference markers for coordinate transformation'
        }
    
    def calculate_relative_coordinates(self, word_coords):
        """
        Convert absolute PDF coordinates to relative coordinates from top-left marker.

        Args:
            word_coords: List of [x1, y1, x2, y2] coordinate sets.

        Returns: 
            List of relative coordinate sets
        """

        # Use top_left marker as reference origin
        ref_x, ref_y = self.markers['top_left']

        relative_coords = []
        for coord_set in word_coords:
            x1, y1, x2, y2 = coord_set
            rel_coords = [
                x1 - ref_x,  # relative x1
                y1 - ref_y,  # relative y1  
                x2 - ref_x,  # relative x2
                y2 - ref_y   # relative y2
            ]
            relative_coords.append(rel_coords)

        return relative_coords
    
    def get_reference_origin(self):
        """Get the reference origin point (top-left marker)."""
        return self.markers['top_left']
    
    
    def validate_marker_positions(self):
        # Configure later  together with layout details in layout_calculation
        """Validate that the markers don't overlap with content area."""
        content_left = self.margin
        content_right = self.page_width - self.margin
        content_top = self.page_height - self.margin
        content_bottom = self.margin

        warnings = []

        for marker_id, (x, y) in self.markers.items():
            # Check if marker is too close to content area
            if abs(x - content_left) < 8*mm or abs(x - content_right) < 8*mm:
                warnings.append(f"Marker {marker_id} may be too close to content area")
            if abs(y - content_top) < 8*mm or abs(y - content_bottom) < 8*mm:
                warnings.append(f" Marker {marker_id} may be too close to content area")

        return warnings

