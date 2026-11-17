// Upload -> local byte ranges; restored results -> evidence-first reading.
const {chromium}=require('playwright');
const assert=require('assert/strict');
const {makePdf,thumbnail}=require('./browser-fixtures.cjs');
const pdf=makePdf(64,100000);
(async()=>{
 const browser=await chromium.launch({headless:true});
 try {
  const page=await browser.newPage({viewport:{width:1280,height:800}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{
   const read=Blob.prototype.arrayBuffer;window.__blobReads=[];
   Blob.prototype.arrayBuffer=function(){window.__blobReads.push(this.size);return read.call(this)};
  });
  let fileDownloads=0;
  await page.route('**/api/**',async route=>{
   const u=new URL(route.request().url()),p=u.pathname;if(!p.startsWith('/api/'))return route.continue();
   if(p.endsWith('/file')){fileDownloads++;return route.fulfill({contentType:'application/pdf',body:pdf});}
   let json={};
   if(p==='/api/pdfs'&&route.request().method()==='POST'){
    await new Promise(r=>setTimeout(r,500));json={pdf_id:'local',filename:'Local.pdf',total_pages:64};
   }else if(p==='/api/tasks')json={tasks:[],total_pages:1,revision:'loading'};
   await route.fulfill({json});
  });
  await page.goto('http://127.0.0.1:5173/');
  await page.locator('input[type=file]').setInputFiles({name:'Local.pdf',mimeType:'application/pdf',buffer:pdf});
  await page.locator('.upload-progress__label').waitFor();
  await page.waitForFunction(()=>document.querySelector('.pdf-pane [data-pdf-page="1"] canvas')?.width>0,{},{timeout:15000}).catch(async e=>{console.log(await page.locator('body').innerText(),errors,await page.evaluate(()=>window.__blobReads));throw e;});
  const reads=await page.evaluate(()=>window.__blobReads);
  assert.equal(fileDownloads,0,'an uploaded PDF must not be downloaded again');
  assert.ok(reads.length>0&&Math.max(...reads)<pdf.length,'local PDF reads must be sliced, not one full arrayBuffer');
  assert.ok(reads.reduce((a,b)=>a+b,0)<pdf.length/2,'opening a large local PDF must stay demand-driven');
  await page.close();

  const restored=await browser.newPage({viewport:{width:1280,height:800}});
  restored.on('pageerror',e=>errors.push(e.message));let bytes=0,bitmaps=0;
  await restored.route('**/api/**',async route=>{
   const u=new URL(route.request().url()),p=u.pathname;if(!p.startsWith('/api/'))return route.continue();
   if(p.endsWith('/file')){bytes++;return route.fulfill({contentType:'application/pdf',body:makePdf(64)});}
   let json={};
   if(p==='/api/pdfs/known')json={pdf_id:'known',filename:'Known.pdf',total_pages:64};
   else if(p==='/api/tasks')json={tasks:[],total_pages:1,revision:'known'};
   else if(p==='/api/tasks/activity')json={task_id:'activity',pdf_id:'known',type:'bioactivity_extraction',status:'completed',progress:1,params:{structure_task_id:'pipe'}};
   else if(p==='/api/tasks/pipe')json={task_id:'pipe',pdf_id:'known',type:'full_pipeline',status:'completed',progress:1,params:{}};
   else if(p.endsWith('/assays')){await new Promise(r=>setTimeout(r,600));json={records:[{COMPOUND_ID:'A',source_pages:'[45]',BIOACTIVITY_OBSERVATIONS:JSON.stringify([{subject_id:'A',page:45,column:'IC50',value:'24'}])}]};}
   else if(p==='/api/tasks/job')json={task_id:'job',pdf_id:'known',type:'structure_extraction',status:'completed',progress:1,params:{pages:[37,45]}};
   else if(p.endsWith('/structures')){await new Promise(r=>setTimeout(r,450));json={records:[{COMPOUND_ID:'A',source_pages:'[37]',BOX_COORDS_FILE:'/box/a.json'},{COMPOUND_ID:'B',PAGE_NUM:45,BOX_COORDS_FILE:'/box/b.json'}],filtered_records:[]};}
   else if(p==='/api/artifacts')json={mime_type:'application/json',content:Buffer.from(JSON.stringify({box:[100,260,280,380],page_width:600,page_height:800})).toString('base64')};
   else if(p.includes('/pages/')||p.includes('/preview')){bitmaps++;json={image:thumbnail};}
   await route.fulfill({json});
  });
  await restored.goto('http://127.0.0.1:5173/?pdf=known&structureTask=job&step=4');
  await restored.waitForFunction(()=>document.querySelector('.pdf-pane')?.dataset.page==='37'&&document.querySelector('[data-pdf-page="37"] canvas')?.width>0);
  assert.equal(await restored.locator('[data-pdf-page="1"] canvas').getAttribute('width'),'0','restoring a job must not render an irrelevant cover first');
  await restored.locator('.review-card .pdf-thumbnail--ready').first().waitFor();
  assert.equal(bytes,1);assert.equal(bitmaps,0);assert.deepEqual(errors,[]);
  console.log(JSON.stringify({uploadDownloads:fileDownloads,localRangeReads:reads.length,localReadBytes:reads.reduce((a,b)=>a+b,0),pdfBytes:pdf.length,completedJobLanding:37,coverRendered:false,sharedPdfDownloads:bytes,errors}));
  bytes=0;
  await restored.goto('http://127.0.0.1:5173/?pdf=known&assayTask=activity&step=4');
  await restored.waitForFunction(()=>document.querySelector('.pdf-pane')?.dataset.page==='45'&&document.querySelector('[data-pdf-page="45"] canvas')?.width>0);
  assert.equal(await restored.locator('[data-pdf-page="37"] canvas').getAttribute('width'),'0','an activity job linked to a pipeline must land on activity evidence');
  assert.equal(bytes,1);assert.deepEqual(errors,[]);
  console.log(JSON.stringify({activityJobLanding:45,linkedPipelineSource:37,activityIntentPreserved:true}));
  await restored.close();
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
