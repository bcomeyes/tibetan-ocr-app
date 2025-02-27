"""
Utilities for extracting embedded images from PDF files.
"""
import io
import os
import PyPDF2
from PIL import Image

def extract_images_from_pdf(pdf_path, output_folder, first_page=1, last_page=None):
    """
    Extract the first image from each page of a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Folder to save extracted images
        first_page (int, optional): First page to process (1-based). Defaults to 1.
        last_page (int, optional): Last page to process. Defaults to None (all pages).
    
    Returns:
        list: List of paths to extracted images
        int: Total number of pages in the PDF
    """
    image_paths = []
    
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        
        # Adjust last_page if not specified
        if last_page is None or last_page > total_pages:
            last_page = total_pages
        
        # Ensure first_page is valid
        if first_page < 1:
            first_page = 1
        
        # Convert to 0-based indexing
        first_page_idx = first_page - 1
        last_page_idx = last_page - 1
        
        # Process each page
        for page_idx in range(first_page_idx, last_page_idx + 1):
            page = reader.pages[page_idx]
            page_num = page_idx + 1  # Convert back to 1-based for filename
            
            # Get all images on the page
            image_found = False
            
            try:
                # Get the resources dictionary
                resources = page.get('/Resources')
                if resources is not None:
                    # Get XObject dictionary
                    xobjects = resources.get('/XObject')
                    if xobjects is not None:
                        # Iterate through all XObjects
                        for obj_name, obj in xobjects.items():
                            try:
                                # Resolve indirect objects
                                if hasattr(obj, 'get_object'):
                                    obj = obj.get_object()
                                
                                # Check if it's an image
                                subtype = None
                                if hasattr(obj, 'get'):
                                    subtype = obj.get('/Subtype')
                                elif isinstance(obj, dict) and '/Subtype' in obj:
                                    subtype = obj['/Subtype']
                                
                                if subtype == '/Image':
                                    try:
                                        # Extract image data
                                        if hasattr(obj, 'get_data'):
                                            image_data = obj.get_data()
                                        else:
                                            # Skip if we can't get image data
                                            continue
                                        
                                        # Determine image format
                                        filter_type = None
                                        filters = None
                                        
                                        # Try to get filters
                                        if hasattr(obj, 'get'):
                                            filters = obj.get('/Filter')
                                        elif isinstance(obj, dict) and '/Filter' in obj:
                                            filters = obj['/Filter']
                                        
                                        if filters is not None:
                                            # Resolve indirect objects
                                            if hasattr(filters, 'get_object'):
                                                filters = filters.get_object()
                                            
                                            if isinstance(filters, list):
                                                filter_type = filters[0]
                                            else:
                                                filter_type = filters
                                        
                                        # Try to determine image format and create image
                                        img = None
                                        ext = None
                                        
                                        if filter_type == '/DCTDecode':
                                            # JPEG image
                                            try:
                                                img = Image.open(io.BytesIO(image_data))
                                                ext = '.jpg'
                                            except Exception as e:
                                                print(f"Error opening JPEG: {e}")
                                                continue
                                                
                                        elif filter_type == '/FlateDecode':
                                            # PNG or other format
                                            try:
                                                # Get color space
                                                color_space = None
                                                if hasattr(obj, 'get'):
                                                    color_space = obj.get('/ColorSpace')
                                                elif isinstance(obj, dict) and '/ColorSpace' in obj:
                                                    color_space = obj['/ColorSpace']
                                                
                                                # Resolve indirect objects
                                                if hasattr(color_space, 'get_object'):
                                                    color_space = color_space.get_object()
                                                
                                                if color_space == '/DeviceRGB':
                                                    # Get width and height
                                                    width = None
                                                    height = None
                                                    
                                                    if hasattr(obj, 'get'):
                                                        width = obj.get('/Width')
                                                        height = obj.get('/Height')
                                                    elif isinstance(obj, dict):
                                                        width = obj.get('/Width')
                                                        height = obj.get('/Height')
                                                    
                                                    # Resolve indirect objects
                                                    if hasattr(width, 'get_object'):
                                                        width = width.get_object()
                                                    if hasattr(height, 'get_object'):
                                                        height = height.get_object()
                                                    
                                                    # Create PIL Image from raw data
                                                    if width is not None and height is not None:
                                                        img = Image.frombytes('RGB', (width, height), image_data)
                                                        ext = '.png'
                                            except Exception as e:
                                                print(f"Error processing PNG: {e}")
                                                continue
                                        
                                        # If we couldn't determine the format, try to open as JPEG
                                        if img is None:
                                            try:
                                                img = Image.open(io.BytesIO(image_data))
                                                ext = '.jpg'
                                            except:
                                                # Try one more time with raw data
                                                try:
                                                    # Get bits per component
                                                    bits = 8
                                                    if hasattr(obj, 'get'):
                                                        bits_val = obj.get('/BitsPerComponent')
                                                        if bits_val is not None:
                                                            bits = bits_val
                                                    
                                                    # Get width and height
                                                    width = None
                                                    height = None
                                                    
                                                    if hasattr(obj, 'get'):
                                                        width = obj.get('/Width')
                                                        height = obj.get('/Height')
                                                    elif isinstance(obj, dict):
                                                        width = obj.get('/Width')
                                                        height = obj.get('/Height')
                                                    
                                                    # Resolve indirect objects
                                                    if hasattr(width, 'get_object'):
                                                        width = width.get_object()
                                                    if hasattr(height, 'get_object'):
                                                        height = height.get_object()
                                                    
                                                    if width is not None and height is not None:
                                                        # Try as grayscale
                                                        img = Image.frombytes('L', (width, height), image_data)
                                                        ext = '.png'
                                                except Exception as e:
                                                    print(f"Failed to create image: {e}")
                                                    continue
                                        
                                        # Save the image if we were able to create it
                                        if img is not None and ext is not None:
                                            image_path = os.path.join(output_folder, f"page_{page_num}{ext}")
                                            img.save(image_path)
                                            image_paths.append(image_path)
                                            
                                            # Only extract the first image from each page
                                            image_found = True
                                            break
                                    except Exception as e:
                                        print(f"Error extracting image data: {e}")
                                        continue
                            except Exception as e:
                                print(f"Error processing XObject: {e}")
                                continue
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
            
            # If no image was found on this page, we'll create a blank image as a placeholder
            if not image_found:
                # Create a blank white image
                blank_img = Image.new('RGB', (800, 1000), color='white')
                image_path = os.path.join(output_folder, f"page_{page_num}_blank.png")
                blank_img.save(image_path)
                image_paths.append(image_path)
    
    return image_paths, total_pages
