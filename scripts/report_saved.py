"""Render the committed measurement ledger as Markdown and a paper-style SVG."""
import argparse,json,math
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET
HEADERS=['Method','Test mIoU','Test OA','Test average recall']
def maxima(panel):return [max((r['values'][j] for _,rs in panel['groups'] for r in rs if r['values'][j] is not None),default=None) for j in range(3)]
def best(v,m):return v is not None and math.isclose(v,m,rel_tol=0,abs_tol=1e-10)
def number(v):return '-' if v is None else f'{v:.2f}'
def svg(panels):
 items=[];y=38
 def text(x,y,s,size=19,weight='normal',anchor='start',italic=False):items.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"'+(' font-style="italic"' if italic else '')+'>'+escape(s)+'</text>')
 def line(x,y,xx,yy):items.append(f'<path d="M{x} {y} L{xx} {yy}" stroke="black" stroke-width="1"/>')
 for panel in panels:
  text(35,y,panel['title'],23);y+=40;top=y-20
  text(45,y,'Method');centers=[870,1030,1190]
  for x,h in zip(centers,HEADERS[1:]):text(x,y,h,18,anchor='middle')
  line(35,y+12,1260,y+12);y+=40;mx=maxima(panel)
  for label,rows in panel['groups']:
   if label:
    y+=10;line(35,y-19,1260,y-19);text(45,y,label,17,italic=True);y+=31
   for r in rows:
    text(45,y,r['label'],18)
    for j,(x,v) in enumerate(zip(centers,r['values'])):
     weight='bold' if best(v,mx[j]) else 'normal';sd=r.get('sd',[None]*3)[j]
     if sd is None:text(x,y,number(v),19,weight,'middle')
     else:
      text(x-5,y,number(v),19,weight,'end');text(x+2,y,f'±{sd:.2f}',14)
    y+=27
  line(775,top,775,y-12);y+=9
  for s in panel['caption']:text(35,y,s,17);y+=24
  y+=38
 text(35,y,'Unmeasured / unfinished tests are not assigned scores.',16);y+=30
 text(35,y,'Layout adapted from Table 2 of Garnot et al. (CVPR 2020). Bold is used only for a column\'s best numeric value.',16);y+=30
 doc=f'<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="{y}" viewBox="0 0 1300 {y}" role="img" aria-label="Research results table"><rect width="1300" height="{y}" fill="white"/><g fill="black" font-family="Times New Roman, Times, serif">'+''.join(items)+'</g></svg>'
 ET.fromstring(doc);return doc,y

def markdown(panels):
    lines=['Research results','','Average recall is the unweighted mean of the 20 class recalls. Bold marks only the best number in each column within each evaluation table.','']
    for panel in panels:
        lines += [panel['title'],'','| '+' | '.join(HEADERS)+' |','|---|---:|---:|---:|']
        mx=maxima(panel)
        for label,rows in panel['groups']:
            if label:lines.append('| '+label+' | | | |')
            for row in rows:
                cells=[]
                for j,v in enumerate(row['values']):
                    cell=number(v)
                    if best(v,mx[j]):cell='**'+cell+'**'
                    sd=row.get('sd',[None]*3)[j]
                    if sd is not None:cell+=f' ± {sd:.2f}'
                    cells.append(cell)
                lines.append('| '+row['label']+' | '+' | '.join(cells)+' |')
        lines+=['']+panel['caption']+['']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',default='records/table_data.json')
    p.add_argument('--output',default='outputs/recorded_results.md')
    p.add_argument('--svg',default='outputs/recorded-results.svg');a=p.parse_args()
    panels=json.loads(Path(a.records).read_text())
    for name,content in [(a.output,markdown(panels)),(a.svg,svg(panels)[0])]:
        out=Path(name);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(content.rstrip()+'\n');print(out)
if __name__=='__main__':main()
