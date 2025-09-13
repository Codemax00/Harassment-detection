import markdown
from pptx import Presentation
from pptx.util import Inches

def markdown_to_pptx(md_file, pptx_file):
    """Convert Marp-style Markdown to a PowerPoint presentation"""
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into slides
    slides_content = content.split('\n---\n')
    
    # Create a presentation object
    prs = Presentation()
    
    # Set slide dimensions for widescreen (16:9)
    prs.slide_width = Inches(16)
    prs.slide_height = Inches(9)
    
    print(f"📄 Found {len(slides_content)} slides to convert...")
    
    for i, slide_md in enumerate(slides_content):
        # Skip Marp configuration header
        if 'marp: true' in slide_md:
            continue
            
        # Add a new slide (Title and Content layout)
        slide_layout = prs.slide_layouts[5]  # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        
        title = slide.shapes.title
        content_box = slide.placeholders[1]
        
        # Split slide into lines
        lines = [line.strip() for line in slide_md.strip().split('\n') if line.strip()]
        
        if not lines:
            continue
        
        # Set title and content
        title.text = lines[0].replace('#', '').strip()
        
        # Add content
        content_text = ""
        for line in lines[1:]:
            # Simple conversion of Markdown elements
            line = line.replace('**', '').replace('*', '  • ')
            content_text += line + '\n'
            
        content_box.text = content_text
        
        print(f"  ✅ Slide {i}: '{title.text}' converted")
        
    # Save the presentation
    prs.save(pptx_file)
    print(f"\n🎉 Presentation saved to: {pptx_file}")

if __name__ == '__main__':
    md_file = 'Project_Overview_Presentation.md'
    pptx_file = 'Safety_Detector_Overview.pptx'
    
    print("🚀 Converting Markdown to PowerPoint...")
    markdown_to_pptx(md_file, pptx_file)
    print("✅ Conversion complete!")
