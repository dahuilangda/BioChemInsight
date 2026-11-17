// Shared layout, stable selection and lazy PDF interactions; isolated HTTP fixtures.
const { chromium } = require('playwright');
const assert = require('assert/strict');
const {makePdf, thumbnail} = require('./browser-fixtures.cjs');
(async()=>{
 require('fs').mkdirSync('/tmp/biochem-layout-review',{recursive:true});
 const base=process.env.WORKBENCH_URL||'http://127.0.0.1:5173';
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:900}});
 await page.addInitScript(()=>{
   const NativeObserver=window.IntersectionObserver;
   window.__intersectionObservers=0;
   window.IntersectionObserver=class extends NativeObserver {
     constructor(callback,options){super(callback,options);window.__intersectionObservers++;}
   };
 });
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 let pdfReads=0,docCount=3,thumbnailReads=0,largeReads=0;
 await page.route('**/api/**',async route=>{
 const u=new URL(route.request().url()),p=u.pathname;
 if(p.includes('/pages/')) {thumbnailReads++;if(u.searchParams.get('max_width')==='1024')largeReads++;}
 if(!p.startsWith('/api/'))return route.continue();
 if(p.endsWith('/file')) {pdfReads++;return route.fulfill({contentType:'application/pdf',body:makePdf(docCount)});}
 let json={};
 if(p==='/api/pdfs/qa')json={pdf_id:'qa',filename:'source.pdf',total_pages:docCount};
 else if(p.includes('/pages/'))json={page:1,image:thumbnail};
 else if(p==='/api/tasks')json={tasks:[],running_count:0,pending_count:0,total_pages:1,revision:'layout'};
 else if(p==='/api/tasks/qa')json={task_id:'qa',type:'structure_extraction',status:'completed',progress:1,params:{},created_at:new Date().toISOString(),updated_at:new Date().toISOString()};
 else if(p.endsWith('/structures'))json={records:[{COMPOUND_ID:'Compound 1',SMILES:'CCO',PAGE_NUM:1}],filtered_records:[]};
 else if(p==='/api/chem/render-batch')json={results:route.request().postDataJSON().items.map(x=>({...x,image:'data:image/png;base64,'+thumbnail}))};
 await route.fulfill({json});});
 await page.goto(`${base}/`);
 const header=await page.locator('.app-header').boundingBox(),nav=await page.locator('.step-nav').boundingBox();
 assert.ok(Math.abs(nav.y-header.y)<5&&nav.x>=header.x+header.width,'upload navigation must share header row');
 await page.screenshot({path:'/tmp/biochem-layout-review/upload.png'});
 await page.goto(`${base}/?pdf=qa&structureTask=qa&step=4`);
 await page.waitForFunction(()=>document.querySelector('.pdf-pane canvas')?.width>0);
 assert.equal(await page.locator('.results-workspace__heading').count(),0);
 assert.equal(await page.getByRole('button',{name:'PDF',exact:true}).count(),0);
 const reviewNav=await page.locator('.step-nav__button').last().boundingBox(),jobs=await page.getByRole('button',{name:'Jobs',exact:true}).boundingBox();assert.ok(reviewNav.x+reviewNav.width<=jobs.x-4,'Review overlaps Jobs');
 const pageField=page.getByRole('textbox',{name:'PDF page',exact:true});await pageField.fill('2');await pageField.press('Enter');
 await page.waitForFunction(()=>document.querySelector('.pdf-pane')?.getAttribute('data-page')==='2'&&!document.querySelector('.pdf-pane__painting'));
 await page.getByRole('button',{name:'Zoom in',exact:true}).click();await page.waitForTimeout(250);
 await page.waitForFunction(()=>!document.querySelector('.pdf-pane__painting'));
 await page.locator('.pdf-pane__viewport').evaluate(node=>{node.scrollTop=node.querySelector('[data-pdf-page="2"]').offsetTop+80;});
 const scrollBefore=await page.locator('.pdf-pane__viewport').evaluate(node=>node.scrollTop);
 const zoomBefore=await page.locator('.pdf-pane__zoom-level').innerText();
 const downloadsBefore=pdfReads;
 await page.evaluate(()=>{
   window.__retainedCanvas=document.querySelector('.pdf-pane canvas');
   window.__retainedPane=document.querySelector('.pdf-pane');
   window.__pdfUnmounts=0;
   window.__pdfObserver=new MutationObserver(records=>{for(const record of records)for(const node of record.removedNodes)if(node===window.__retainedCanvas||node===window.__retainedPane||node.contains?.(window.__retainedCanvas))window.__pdfUnmounts++;});
   window.__pdfObserver.observe(document.querySelector('.workbench-document'),{childList:true,subtree:true});
 });
 const paintsBefore=Number(await page.locator('.pdf-pane').getAttribute('data-render-count'));
 const pdfBefore=await page.locator('.pdf-pane').boundingBox();
 for(const name of ['Upload PDF','Structures','Activity & ADME','Review','Structures','Review']) {
   await page.locator('.step-nav').getByRole('button',{name,exact:true}).click();
   await page.waitForTimeout(180);
   assert.ok(await page.evaluate(()=>window.__retainedCanvas===document.querySelector('.pdf-pane canvas')),'navigation remounted canvas');
   assert.equal(Number(await page.locator('.pdf-pane').getAttribute('data-render-count')),paintsBefore,'navigation repainted PDF');
   assert.equal(await pageField.inputValue(),'2','navigation reset current page');
   assert.equal(await page.locator('.pdf-pane__zoom-level').innerText(),zoomBefore,'navigation reset zoom');
   assert.equal(await page.locator('.pdf-pane__viewport').evaluate(node=>node.scrollTop),scrollBefore,'navigation reset scroll');
   assert.equal(pdfReads,downloadsBefore,'navigation refetched PDF');
   const bounds=await page.locator('.pdf-pane').boundingBox();for(const key of ['x','y','width','height'])assert.ok(Math.abs(bounds[key]-pdfBefore[key])<.5,'navigation changed PDF geometry: '+key);
 }
 assert.equal(await page.evaluate(()=>window.__pdfUnmounts),0,'navigation removed PDF DOM');
 await page.evaluate(()=>window.__pdfObserver.disconnect());
 const pdf=await page.locator('.pdf-pane').boundingBox(),toolbar=await page.locator('.review-toolbar').boundingBox();
 assert.ok(toolbar.x>=pdf.x+pdf.width,'result switching must be in right pane');
 const before=pdfReads;
 await page.getByRole('button',{name:'Table',exact:true}).click();
 await page.locator('.review-table').waitFor();assert.ok(await page.locator('.pdf-pane').isVisible());
 await page.getByRole('button',{name:'Cards',exact:true}).click();
 await page.locator('.review-card').first().waitFor();assert.ok(await page.locator('.pdf-pane').isVisible());
 assert.equal(pdfReads,before,'switch must preserve mounted PDF');
 await page.screenshot({path:'/tmp/biochem-layout-review/review-desktop.png'});
 for(const width of [768,375,320]){
 await page.setViewportSize({width,height:900});
 await page.waitForTimeout(300);
 assert.ok(await page.locator('.pdf-pane').isVisible());
 assert.ok(await page.locator('.review-split__list').isVisible());
 await page.getByRole('button',{name:'Table',exact:true}).click();
 assert.ok(await page.locator('.review-table').isVisible());
 await page.getByRole('button',{name:'Cards',exact:true}).click();
 assert.ok(await page.locator('.review-card').first().isVisible());
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'no page overflow');
 const bounds=await page.locator('.review-toolbar').boundingBox();
 for(const item of await page.locator('.review-toolbar__controls > button, .review-pagination, .review-toolbar__views').all()){const box=await item.boundingBox();assert.ok(box.x+box.width<=bounds.x+bounds.width+1,'toolbar clipped');assert.ok(box.y>=bounds.y&&box.y+box.height<=bounds.y+bounds.height+1,'toolbar wrapped');}
 await page.getByRole('button',{name:'Search results',exact:true}).click();await page.getByRole('searchbox',{name:'Search results',exact:true}).fill('missing');await page.keyboard.press('Escape');
 assert.ok(await page.locator('.pdf-pane').isVisible());
 await page.getByRole('button',{name:'Search results',exact:true}).click();await page.getByRole('searchbox',{name:'Search results',exact:true}).fill('');await page.keyboard.press('Escape');
 await page.screenshot({path:`/tmp/biochem-layout-review/review-${width}.png`});
 }

 await page.setViewportSize({width:1280,height:900});
 await page.goto(`${base}/?pdf=qa&step=1`);
 await page.waitForFunction(()=>document.querySelector('.workbench-document .pdf-pane canvas')?.width>0);
 const preview=await page.locator('.workbench-document').boundingBox(),imports=await page.locator('.upload-controls').boundingBox();
 assert.ok(preview.x+preview.width<=imports.x);
 await page.screenshot({path:'/tmp/biochem-layout-review/upload-loaded.png'});
 const splitter=page.getByRole('separator',{name:'Resize PDF pane'});const oldPreviewWidth=(await page.locator('.workbench-document').boundingBox()).width;
 await splitter.focus();await page.keyboard.press('ArrowLeft');assert.ok((await page.locator('.workbench-document').boundingBox()).width<oldPreviewWidth);
 await splitter.dblclick();
 assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('bci:workbench:split'))),55);
 for(const step of [2,3]) {
 await page.goto(`${base}/?pdf=qa&step=${step}`);
 const card=page.locator('.page-card').first();await card.waitFor();
 await card.locator('.pdf-thumbnail--ready').waitFor();
 const footer=card.locator('.page-card__footer');
 const before=await card.boundingBox(),footerBefore=await footer.boundingBox(),labelBefore=await footer.locator('.page-card__label').boundingBox();
 for(let n=0;n<4;n++) {
 await footer.locator('.page-card__label').click();
 const after=await card.boundingBox(),footerAfter=await footer.boundingBox(),labelAfter=await footer.locator('.page-card__label').boundingBox();
 assert.ok(Math.abs(labelAfter.x-labelBefore.x)<.5,'page number moved');
 assert.ok(Math.abs(labelAfter.x+labelAfter.width/2-footerAfter.x-footerAfter.width/2)<.5,'page number not centered');
 for(const key of ['x','y','width','height']) {assert.ok(Math.abs(after[key]-before[key])<.5,'card geometry changed '+key);assert.ok(Math.abs(footerAfter[key]-footerBefore[key])<.5,'footer geometry changed '+key);}
 }
 const zoom=footer.locator('.magnify-btn');assert.equal(await zoom.count(),1);
 assert.ok((await zoom.boundingBox()).x<(await footer.locator('.page-card__label').boundingBox()).x);
 await page.screenshot({path:`/tmp/biochem-layout-review/stage-${step}.png`});
 const selectionBefore=await card.getAttribute('aria-pressed');
 await zoom.click();await page.waitForFunction(()=>document.querySelector('.workbench-document .pdf-pane canvas')?.width>0);
 assert.equal(await card.getAttribute('aria-pressed'),selectionBefore,'preview changed selection');
 assert.equal(await page.getByRole('dialog',{name:'Image preview',exact:true}).count(),0);
 await page.screenshot({path:`/tmp/biochem-layout-review/stage-${step}-preview.png`});
 const stagePdf=await page.locator('.workbench-document').boundingBox(),stageData=await page.locator('.stage-data').boundingBox();assert.ok(stagePdf.x+stagePdf.width<=stageData.x,'stage PDF must remain on left');
 }
 await page.goto(`${base}/?pdf=qa&structureTask=qa&step=2`);
 await page.locator('.step-nav').getByRole('button',{name:'Structures',exact:true}).click();
 const selected=page.locator('.page-card').first();await selected.locator('.pdf-thumbnail--ready').waitFor();
 await page.locator('.step-nav').getByRole('button',{name:'Structures',exact:true}).click();
 const startBox=await selected.boundingBox();if(await selected.getAttribute('aria-pressed')!=='true')await selected.locator('.page-card__label').click();
 await page.locator('.step-nav').getByRole('button',{name:'Activity & ADME',exact:true}).click();
 await selected.locator('.page-card__label').click();
 assert.ok((await selected.getAttribute('class')).includes('structure-selected'));
 assert.ok((await selected.getAttribute('class')).includes('assay-selected'));
 const endBox=await selected.boundingBox();for(const key of ['x','y','width','height'])assert.ok(Math.abs(startBox[key]-endBox[key])<.5,'dual selection changed layout');
 await page.getByRole('button',{name:'Jobs',exact:true}).click();await page.locator('.jobs-panel').waitFor();
 const jobButton=await page.getByRole('button',{name:'Jobs',exact:true}).boundingBox(),drawer=await page.locator('.jobs-panel').boundingBox();
 assert.ok(Math.abs(jobButton.x+jobButton.width-drawer.x-drawer.width)<5,'Jobs drawer anchor differs');
 await page.screenshot({path:'/tmp/biochem-layout-review/jobs.png'});
 docCount=500;const previousThumbnails=thumbnailReads;
 await page.goto(`${base}/?pdf=qa&step=2`);
 await page.waitForFunction(()=>document.querySelector('.workbench-document .pdf-pane canvas')?.width>0);
 await page.locator('.page-card').first().locator('.pdf-thumbnail--ready').waitFor();
 await page.getByRole('button',{name:'Enlarge page 1',exact:true}).click();
 const pdfInput=page.getByRole('textbox',{name:'PDF page',exact:true});await pdfInput.fill('450');await pdfInput.press('Enter');
 await page.waitForFunction(()=>document.querySelector('.workbench-document .pdf-pane')?.getAttribute('data-page')==='450'&&!document.querySelector('.pdf-pane__painting'));
 const intersectionObservers=await page.evaluate(()=>window.__intersectionObservers);
 assert.ok(intersectionObservers<10,'long documents must share visibility observers');
 assert.ok(thumbnailReads-previousThumbnails<100,'long document eagerly fetched thumbnail pages');
 assert.equal(largeReads,0,'page cards must not request full-size PNGs');
 assert.ok(await page.locator('.pdf-pane canvas').evaluateAll(cs=>cs.filter(c=>c.width>0).length)<=4,'PDF must retain display buffers only for nearby pages');
 assert.deepEqual(errors,[]);console.log(JSON.stringify({passed:true,pdfReads,navigationRemounts:0,navigationRepaints:0,preservesPageZoomScroll:true,longDocumentPages:500,intersectionObservers,desktop:1440,mobile:[768,375,320]}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
