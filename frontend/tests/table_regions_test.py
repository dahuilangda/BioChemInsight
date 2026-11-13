import tempfile
import unittest
from pathlib import Path
import fitz
from utils.table_regions import table_windows

class TableRegionsTest(unittest.TestCase):
    def test_repeated_blocks_are_split_without_losing_source_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            document=fitz.open();page=document.new_page(width=600,height=800)
            left,top,width,step=40,100,520,18
            for column in range(9):page.draw_line((left+column*width/8,top),(left+column*width/8,top+step*29),width=.7)
            for row in range(30):page.draw_line((left,top+row*step),(left+width,top+row*step),width=.7)
            for column in range(8):page.insert_text((left+column*width/8+3,top+12),'ID' if column%2==0 else 'Value',fontsize=8)
            for row in range(28):
                for group in range(4):page.insert_text((left+group*width/4+3,top+step*(row+1)+12),str(group*28+row+1),fontsize=8)
            image=Path(temporary)/'page.png';page.get_pixmap(dpi=180).save(image)
            windows=table_windows(page,image,Path(temporary)/'windows')
            self.assertTrue(windows)
            for group in range(4):
                selected=[item for item in windows if item['group_index']==group]
                self.assertEqual(sum(item['row_end']-item['row_start'] for item in selected),28)
                self.assertEqual(selected[0]['row_start'],1)
                self.assertEqual(selected[-1]['row_end'],29)
            document.close()
    def test_plain_page_does_not_become_a_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            document=fitz.open();page=document.new_page();page.insert_text((40,80),'Document prose without a table.')
            image=Path(temporary)/'page.png';page.get_pixmap().save(image)
            self.assertEqual(table_windows(page,image,Path(temporary)/'windows'),[])
            document.close()
if __name__=='__main__':unittest.main()
