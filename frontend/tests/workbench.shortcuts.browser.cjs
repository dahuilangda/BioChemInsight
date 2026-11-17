// Large result fixture + real PDF.js; parsing and persistence are isolated HTTP fixtures.
const {chromium}=require('playwright');const assert=require('assert/strict');
const {makePdf,thumbnail}=require('./browser-fixtures.cjs');const fs=require('fs');
(async()=>{const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1600,height:1000}});page.setDefaultTimeout(12000);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 let records=Array.from({length:2500},(_,i)=>({RECORD_ID:`row-${i}`,COMPOUND_ID:`C-${String(i).padStart(4,'0')}`,SMILES:'CCO',PAGE_NUM:i%64+1,'IC50 (µM)':String(i+1),USER_MEASUREMENT_EDITS:JSON.stringify({'IC50 (µM)':String(i+1)})}));
 let revision=0,writes=0,parses=[],failParse=true,releaseStructure;
 const task=()=>({task_id:'large-review',type:'structure_extraction',status:'completed',pdf_id:'qa',params:{records_revision:String(revision)}});
 await page.route('**/api/**',async route=>{const path=new URL(route.request().url()).pathname;let json={};
  if(path.endsWith('/file'))return route.fulfill({contentType:'application/pdf',body:makePdf(64)});
  if(path==='/api/pdfs/qa')json={pdf_id:'qa',filename:'64-page interaction fixture.pdf',total_pages:64};
  else if(path==='/api/tasks/large-review')json=task();
  else if(path.endsWith('/structures')){if(route.request().method()==='PUT'){const body=route.request().postDataJSON();assert.equal(body.base_revision,String(revision));records=body.records;revision++;writes++;}json={task:task(),records,filtered_records:[]};}
  else if(path==='/api/tasks')json={tasks:[task()],total_pages:1,revision:'large'};
  else if(path.includes('/pages/')||path.includes('/preview'))json={image:thumbnail};
  else if(path==='/api/chem/render-batch')json={results:route.request().postDataJSON().items.map(item=>({...item,image:`data:image/png;base64,${thumbnail}`}))};
  else if(path.endsWith('/regions/measurements')){parses.push(route.request().postDataJSON());if(failParse){failParse=false;return route.fulfill({status:503,json:{detail:'Temporary parsing failure'}});}json={records:[{COMPOUND_ID:'C-0002',BIOACTIVITY_OBSERVATIONS:JSON.stringify([{subject_id:'C-0002',column:'Permeability (cm/s)',value:'1.4e-6',association_status:'verified',page:1}])}],gaps:[]};}
  else if(path.endsWith('/regions/recognize')) {await new Promise(resolve => {releaseStructure=resolve;});json={SMILES:'CCN',COMPOUND_ID:'',PAGE_NUM:1};}
  else if(path.endsWith('/history'))json={revisions:[],current_revision:String(revision)};
  return route.fulfill({json});
 });
 const start=Date.now();await page.goto(`${process.env.WORKBENCH_URL||'http://127.0.0.1:5173'}/?pdf=qa&structureTask=large-review&step=4`);
 await page.getByRole('button',{name:'Table',exact:true}).click();await page.locator('tbody tr[data-row-index]').first().waitFor();
 const loadMs=Date.now()-start;assert.ok(await page.locator('tbody tr[data-row-index]').count()<=200,'results must be paginated');
 await page.keyboard.press('Control+/');await page.getByRole('dialog',{name:'Keyboard shortcuts'}).waitFor();await page.keyboard.press('Escape');
 await page.keyboard.press('Control+f');const search=page.getByRole('searchbox',{name:'Search results',exact:true});await search.fill('C-2499');
 await page.waitForFunction(()=>document.querySelectorAll('tbody tr[data-row-index]').length===1);assert.equal(await page.locator('tbody .page-cell__id-input').inputValue(),'C-2499');
 await search.fill('b');assert.equal(await page.locator('.pdf-pane').getAttribute('data-drawing'),'false');await search.fill('');await search.press('Escape');
 await page.keyboard.press('Control+Shift+s');await page.waitForFunction(()=>document.activeElement?.id==='review-sort-column');await page.keyboard.press('Escape');
 const first=page.locator('tbody .page-cell__id-input').first();await first.fill('canceled');await first.press('Escape');assert.equal(writes,0);assert.equal(await first.inputValue(),'C-0000');
 await first.fill('Saved ID');await first.press('Enter');await page.waitForFunction(()=>document.querySelector('[aria-label="Saved"]'));
 assert.equal(records[0].COMPOUND_ID,'Saved ID');assert.equal(writes,1);
 const dose=page.locator('tbody tr[data-row-index]').first().locator('.review-value-input').first();await dose.fill('7.5');await dose.press('Control+s');await page.waitForFunction(()=>document.querySelector('[aria-label="Saved"]'));assert.equal(JSON.parse(records[0].USER_MEASUREMENT_EDITS)['IC50 (µM)'],'7.5');

 // Pressing B in a text editor must remain typing.
 await first.focus();await first.press('b');assert.equal(await page.locator('.pdf-pane').getAttribute('data-drawing'),'false');await first.press('Escape');
 await page.locator('.pdf-pane__viewport').focus();await page.keyboard.press('b');assert.equal(await page.locator('.pdf-pane').getAttribute('data-drawing'),'true');
 const paper=page.locator('.pdf-pane [data-pdf-page="1"] .pdf-pane__paper');await paper.scrollIntoViewIfNeeded();
 const bounds=await paper.boundingBox();const point=(x,y)=>({x:bounds.x+x*bounds.width,y:bounds.y+y*bounds.height});
 const a=point(.12,.16),b=point(.4,.33);await page.mouse.move(a.x,a.y);await page.mouse.down();await page.mouse.move(b.x,b.y,{steps:5});await page.mouse.up();
 const selection=page.getByLabel('Your PDF selection',{exact:true});await selection.waitFor();
 const read=()=>selection.evaluate(e=>({x:parseFloat(e.style.left)/100,y:parseFloat(e.style.top)/100,w:parseFloat(e.style.width)/100,h:parseFloat(e.style.height)/100}));
 let box=await read();assert.ok(Math.abs(box.x-.12)<.003&&Math.abs(box.w-.28)<.003);
 const corner=await page.locator('.pdf-region-handle--se').boundingBox();await page.mouse.move(corner.x+corner.width/2,corner.y+corner.height/2);await page.mouse.down();await page.mouse.move(corner.x+corner.width/2+18,corner.y+corner.height/2+12,{steps:4});await page.mouse.up();
 let resized=await read();assert.ok(resized.w>box.w&&resized.h>box.h);
 const sb=await selection.boundingBox();await page.mouse.move(sb.x+sb.width/2,sb.y+sb.height/2);await page.mouse.down();await page.mouse.move(sb.x+sb.width/2+14,sb.y+sb.height/2+9,{steps:4});await page.mouse.up();
 let moved=await read();assert.ok(moved.x>resized.x&&moved.y>resized.y);assert.ok(Math.abs(moved.w-resized.w)<.001);
 await page.keyboard.press('ArrowRight');await page.keyboard.press('Shift+ArrowDown');let nudged=await read();assert.ok(nudged.x>moved.x&&nudged.y>moved.y);
 assert.equal(parses.length,0,'drawing and adjusting must not parse');
 // Ctrl Enter works after choosing a purpose, including while the select is focused.
 await page.getByRole('combobox',{name:'Selection purpose'}).selectOption('table');await page.getByRole('combobox',{name:'Selection purpose'}).focus();await page.keyboard.press('Control+Enter');
 await page.locator('.pdf-region-actions [role="alert"]').waitFor();assert.equal(parses.length,1);assert.equal(await selection.count(),1,'failed parsing keeps the box for retry');
 await page.keyboard.press('Control+Enter');await page.locator('.pdf-region-actions').waitFor({state:'detached'});
 assert.equal(parses.length,2);assert.equal(parses[1].page,1);assert.ok(Math.abs(parses[1].x-nudged.x)<.003);
 await page.waitForFunction(()=>document.querySelector('[aria-label="Saved"]'));
 assert.equal(records.length,2500,'a uniquely identified table result attaches to its record');assert.ok(records[2].BIOACTIVITY_OBSERVATIONS.includes('Permeability'));
 // A slow parse must not overwrite a different row after the selected one is deleted.
 await first.focus();await page.locator('.pdf-pane__viewport').focus();await page.keyboard.press('b');
 const latestBounds=await paper.boundingBox();const pa={x:latestBounds.x+.12*latestBounds.width,y:latestBounds.y+.16*latestBounds.height};
 await page.mouse.move(pa.x,pa.y);await page.mouse.down();await page.mouse.move(pa.x+100,pa.y+70,{steps:3});await page.mouse.up();
 await page.getByRole('combobox',{name:'Selection purpose'}).selectOption('structure');
 await page.getByRole('checkbox',{name:'Replace structure for Saved ID',exact:true}).check();
 const request=page.waitForRequest(r=>r.url().endsWith('/regions/recognize'));await page.getByRole('button',{name:'Recognize',exact:true}).click();await request;
 page.once('dialog',d=>d.accept());await page.locator('tbody tr[data-row-index]').first().getByRole('button',{name:'Delete',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('[aria-label="Saved"]'));
 assert.equal(records.length,2499);releaseStructure();
 await page.getByText('The selected record was removed while parsing. Draw a new box to continue.',{exact:true}).waitFor();
 assert.equal(records[0].SMILES,'CCO');assert.equal(await page.getByRole('dialog',{name:'Edit structure',exact:true}).count(),0);
 await page.locator('.pdf-pane__viewport').focus();await page.keyboard.press('Escape');
 // A canceled selection never writes; help remains usable on phones.
 await page.locator('.pdf-pane__viewport').focus();await page.keyboard.press('b');await page.keyboard.press('Escape');assert.equal(await page.locator('.pdf-pane').getAttribute('data-drawing'),'false');
 if(await page.getByRole('button',{name:'Got it',exact:true}).count())await page.getByRole('button',{name:'Got it',exact:true}).click();
 await page.setViewportSize({width:390,height:844});await page.getByRole('button',{name:'Keyboard shortcuts',exact:true}).click();const help=page.getByRole('dialog',{name:'Keyboard shortcuts'});await help.waitFor();const hb=await help.boundingBox();assert.ok(hb.x>=0&&hb.x+hb.width<=391);
 assert.ok(await help.evaluate(e=>{const r=e.getBoundingClientRect();return e.contains(document.elementFromPoint(r.left+50,r.top+110));}),'PDF controls must not cover the shortcut dialog');
 const dir=process.env.QA_OUTPUT_DIR;if(dir){fs.mkdirSync(dir,{recursive:true});await page.screenshot({path:`${dir}/keyboard-help-mobile.png`});}
 await help.getByRole('button',{name:'Close keyboard shortcuts',exact:true}).click();
 // Native Chromium touch events exercise the same pointer path used on a phone.
 const cdp=await page.context().newCDPSession(page);await cdp.send('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:1});
 await page.locator('.workbench-document__toggle').click();assert.equal(await page.locator('.workbench-document__toggle').getAttribute('aria-expanded'),'false');await page.keyboard.press('b');await page.waitForFunction(()=>document.querySelector('.pdf-pane')?.dataset.drawing==='true');assert.equal(await page.locator('.workbench-document__toggle').getAttribute('aria-expanded'),'true');
 const sourcePage=await page.locator('.pdf-pane').getAttribute('data-page');
 const touchPaper=await page.locator(`.pdf-pane [data-pdf-page="${sourcePage}"] .pdf-pane__paper`).boundingBox();
 const view=await page.locator('.pdf-pane__viewport').boundingBox();
 assert.ok(view.height>=180,'phone drawing must leave a usable PDF canvas');
 const tx=Math.max(touchPaper.x,view.x)+20,ty=Math.max(touchPaper.y,view.y)+12;
 const ty2=Math.min(ty+35,view.y+view.height-8,touchPaper.y+touchPaper.height-8);
 assert.ok(ty2>ty+5,'mobile PDF must leave room for a touch selection');
 await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:tx,y:ty}]});
 await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:tx+70,y:ty2}]});
 await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
 await page.getByLabel('Your PDF selection',{exact:true}).waitFor();
 if(dir)await page.screenshot({path:`${dir}/draw-box-mobile.png`});
 await page.locator('.pdf-region-actions').getByRole('button',{name:'Cancel',exact:true}).click();
 assert.equal(parses.length,2,'touch drawing and cancellation must not parse');
 assert.deepEqual(errors,[]);console.log(JSON.stringify({records:2500,pdfPages:64,loadMs,paginated:true,typingProtected:true,escapeCancelsEdit:true,shortcutHelp:true,search:true,sort:true,boxDraw:true,boxResize:true,boxMove:true,boxNudge:true,parseRetry:true,tableAssociation:true,mobileHelp:true,nativeTouchDraw:true,drawReopensPdf:true,saveShortcut:true,deletedTargetProtected:true,writes,parses:parses.length}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
