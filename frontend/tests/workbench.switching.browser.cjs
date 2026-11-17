const {chromium}=require('playwright');
const assert=require('assert/strict');
const {makePdf,thumbnail}=require('./browser-fixtures.cjs');
(async()=>{
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1280,height:800}});
 const errors=[],started={};let fileReads=0,chemBatches=0;const pdfReads={};const taskReads={};
 page.on('pageerror',e=>errors.push(e.message));
 const job=(id,pdf='shared')=>({task_id:id,pdf_id:pdf,type:'structure_extraction',status:'completed',progress:1,result_path:'/result.csv',params:{},created_at:new Date().toISOString(),updated_at:new Date().toISOString()});
 const jobs=[job('A'),job('B','other'),job('C'),job('D','failed')];
 await page.route('**/api/**',async route=>{
  const u=new URL(route.request().url()),p=u.pathname;if(!p.startsWith('/api/'))return route.continue();
  if(p.endsWith('/file')){fileReads++;const id=p.split('/')[3];pdfReads[id]=(pdfReads[id]||0)+1;return route.fulfill({contentType:'application/pdf',body:makePdf(16)});}
  let json={};
  if(p==='/api/tasks')json={tasks:jobs,total_pages:1,revision:'switching'};
  else if(p.startsWith('/api/pdfs/')){
   started[p]=Date.now();if(p.includes('other'))await new Promise(r=>setTimeout(r,300));
   json={pdf_id:p.split('/').pop(),filename:'Shared.pdf',total_pages:16};
  }else if(p.endsWith('/structures')){
   started[p]=Date.now();const id=p.split('/')[3];
   if(id==='B')await new Promise(r=>setTimeout(r,800));
   if(id==='C')await new Promise(r=>setTimeout(r,120));
   if(id==='D')return route.fulfill({status:500,json:{detail:'Fixture result unavailable'}});
   json={records:[{COMPOUND_ID:id,PAGE_NUM:id==='C'?8:5,SMILES:'CCO',BOX_COORDS_FILE:'/box.json'}],filtered_records:[]};
  }else if(p.startsWith('/api/tasks/')){const id=p.split('/').pop();taskReads[id]=(taskReads[id]||0)+1;json=jobs.find(j=>j.task_id===id)||{};}
  else if(p==='/api/artifacts')json={mime_type:'application/json',content:Buffer.from(JSON.stringify({box:[100,260,280,380],page_width:600,page_height:800})).toString('base64')};
  else if(p==='/api/chem/render-batch'){chemBatches++;json={results:route.request().postDataJSON().items.map(x=>({...x,image:'data:image/png;base64,'+thumbnail}))};}
  await route.fulfill({json});
 });
 const ready=n=>page.waitForFunction(n=>document.querySelector('.pdf-pane')?.dataset.page===String(n)&&document.querySelector(`[data-pdf-page="${n}"] canvas`)?.width>0,n);
 try{
  await page.goto('http://127.0.0.1:5173/?pdf=shared&structureTask=A&step=4');await ready(5);
  await page.locator('.review-card').first().getByAltText('Extracted structure',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Jobs',exact:true}).click();
  const before=await page.locator('.pdf-pane').boundingBox();
  await page.evaluate(()=>window.__reader=document.querySelector('.pdf-pane'));
  const preparingB=page.waitForRequest(request=>new URL(request.url()).pathname==='/api/pdfs/other/file');
  await page.getByRole('button',{name:'Open job B',exact:true}).click();
  await page.locator('.job-opening-note').waitFor();
  assert.ok(await page.evaluate(()=>window.__reader===document.querySelector('.pdf-pane')),'opening must retain the existing reader');
  assert.equal(await page.locator('.review-card .page-cell__id-input').first().inputValue(),'A','old results stay intact until commit');
  const during=await page.locator('.pdf-pane').boundingBox();for(const key of ['x','y','width','height'])assert.equal(during[key],before[key],'no geometry shift while opening');
  await preparingB;
  await page.getByRole('button',{name:'Open job C',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.review-card .page-cell__id-input')?.value==='C');await ready(8);
  await page.waitForTimeout(900);
  assert.equal(await page.locator('.review-card .page-cell__id-input').first().inputValue(),'C','a late B result must not overwrite C');
  assert.ok(Math.abs(started['/api/pdfs/other']-started['/api/tasks/B/structures'])<100,'metadata and results start in parallel');
  await page.getByRole('button',{name:'Jobs',exact:true}).click();
  await page.getByRole('button',{name:'Open job D',exact:true}).click();
  await page.getByText('Fixture result unavailable',{exact:true}).waitFor();
  assert.equal(await page.locator('.review-card .page-cell__id-input').first().inputValue(),'C','failed open preserves the current result');
  assert.equal(await page.locator('.pdf-pane__open').getAttribute('href'),'/api/pdfs/shared/file');
  assert.equal(pdfReads.shared,1,'same-PDF jobs share their parsed document');
  assert.equal(pdfReads.other,1,'the requested PDF starts preparing alongside results');
  assert.equal(chemBatches,1,'identical molecules reuse the RDKit preview');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({atomicSwitch:true,staleResultIgnored:true,failedOpenPreservesWorkspace:true,pdfDownloads:pdfReads,rdkitBatches:chemBatches}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
