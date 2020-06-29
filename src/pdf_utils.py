
import PyPDF2
import StringIO
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont, TTFError

def merge_locus_pdf(out_dir, gene_name, write_path, canvas_size=(175, 135)):

    # register font
    try:
        font_name = "OpenSans-Regular"
        pdfmetrics.registerFont(TTFont(font_name, 'OpenSans-Regular.ttf'))
    except TTFError:
        # could not load font, use default
        font_name = 'Helvetica'
        pass

    typhoon_page = PyPDF2.PdfFileReader('%s/typhoon/%s.pdf' % (out_dir, gene_name)).getPage(0)
    cc_page = PyPDF2.PdfFileReader('%s/cc/%s.pdf' % (out_dir, gene_name)).getPage(0)
    lines_page = PyPDF2.PdfFileReader('%s/lines/%s.pdf' % (out_dir, gene_name)).getPage(0)

    canvas_writer = PyPDF2.PdfFileWriter()
    canvas_page = canvas_writer.addBlankPage(*canvas_size)

    canvas_page.mergeScaledTranslatedPage(typhoon_page, 0.08, 10, 0)
    canvas_page.mergeScaledTranslatedPage(cc_page, 0.19, 65, 92)
    canvas_page.mergeScaledTranslatedPage(lines_page, 0.175, 60, 5)

    def write_fig_sublabel(text, loc, canvas_page):

        # load and set font
        packet = StringIO.StringIO()
        cv = canvas.Canvas(packet, pagesize=(20, 20))
        cv.setFont(font_name, 7)

        # write to temp PDF
        cv.drawString(0, 0, text)
        cv.save()

        # write contents to canvas PDF
        packet.seek(0)
        a_page = PyPDF2.PdfFileReader(packet).getPage(0)
        canvas_page.mergeScaledTranslatedPage(a_page, 1, loc[0], loc[1])

    write_fig_sublabel('A', (5, 127), canvas_page)
    write_fig_sublabel('B', (63, 127), canvas_page)
    write_fig_sublabel('C', (63, 85), canvas_page)
    
    with open(write_path, 'wb') as write_file:
        canvas_writer.write(write_file)

    