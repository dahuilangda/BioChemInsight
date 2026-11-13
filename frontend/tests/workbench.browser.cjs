const { makePdf, thumbnail } = require('./browser-fixtures.cjs');
const fixturePdf = makePdf(1);
// Layout regression checks use isolated API fixtures, not model inference.
const {chromium}=require('playwright');const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true});const page=await browser.newPage();
 page.setDefaultTimeout(10000);
 const image='data:image/svg+xml;base64,'+Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="220" height="170"><path d="M45 40 L90 65 L90 115 L45 140 L5 115 L5 65 Z M90 65 L140 40 L180 65" fill="none" stroke="black" stroke-width="2"/></svg>').toString('base64');
 let recordCount=1;let submittedRecords=null;let artifactRequests=0;let failArtifactOnce=true;
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/**',async r=>{
 const u=new URL(r.request().url());const p=u.pathname;if(!p.startsWith('/api/'))return r.continue();
 let json={};
 if(p==='/api/tasks')json={tasks:[],running_count:0,pending_count:0,total_count:0,total_pages:1,revision:'qa'};
 else if(p==='/api/pdfs/qa')json={pdf_id:'qa',filename:'Patent evidence.pdf',total_pages:1};
 else if(p.endsWith('/file'))return r.fulfill({contentType:'application/pdf',body:fixturePdf});
 else if(p.includes('/pages/'))json={page:1,image:thumbnail};
 else if(p.endsWith('/structures')&&r.request().method()==='PUT'){submittedRecords=r.request().postDataJSON().records;json={records:submittedRecords,filtered_records:[]};}
 else if(p.endsWith('/structures'))json={records:Array.from({length:recordCount},(_,i)=>({COMPOUND_ID:String(53+i),SMILES:'C'.repeat(i+2),PAGE_NUM:1,SEGMENT_FILE:`/qa/original-${i}.svg`})),filtered_records:[]};
 else if(p==='/api/artifacts'){artifactRequests++;if(failArtifactOnce){failArtifactOnce=false;return r.fulfill({status:404,json:{detail:'Source not available'}});}json={mime_type:'image/svg+xml',content:image.split(',')[1]};}
 else if(p.endsWith('/assays'))json={records:[{COMPOUND_ID:'53','IC50 (nM)':12.5}]};
 else if(p==='/api/chem/render-batch')json={results:r.request().postDataJSON().items.map(i=>({key:i.key,smiles:i.smiles,image}))};
 else if(p==='/api/chem/render')json={smiles:'CCO',image};
 else if(p==='/api/chem/jsme-molblock')json={molblock:r.request().postDataJSON().molblock,changed:false};
 else if(p.startsWith('/api/tasks/'))json={task_id:p.split('/')[3],type:p.includes('assay')?'assay_extraction':'structure_extraction',status:'completed',pdf_id:'qa',progress:1,params:{pages:'1'},created_at:new Date().toISOString(),updated_at:new Date().toISOString()};
 else console.log('unhandled',p);
 await r.fulfill({json});
 });
 for(const w of [320,375,768,1280,1920]){
 await page.setViewportSize({width:w,height:1000});await page.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&structureTask=structure-qa&assayTask=assay-qa&step=4`);
 await page.locator('.review-card').first().waitFor();
 await page.getByAltText('Extracted structure',{exact:true}).first().waitFor();
 if(w===320){await page.getByRole('button',{name:/Retry original/}).click();}
 await page.getByAltText('Source structure',{exact:true}).first().waitFor();
 if(w===320){
  const original=page.locator('.review-card__enlarge').first();
  await original.click();
  const preview=page.getByRole('dialog',{name:'Image preview',exact:true});await preview.waitFor();
  if(await preview.locator('.floating-panel').count())throw Error('Preview unexpectedly opened an editor');
  await page.keyboard.press('Escape');await preview.waitFor({state:'hidden'});
  if(!await original.evaluate(element=>document.activeElement===element))throw Error('Preview did not restore keyboard focus');
  await page.getByRole('button',{name:'Open source page 1',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.pdf-pane')?.getAttribute('data-page')==='1'&&document.querySelector('.pdf-pane canvas')?.width>0);
  await page.getByRole('button',{name:'Cards',exact:true}).click();
 }

 await page.locator('.review-card dt').filter({hasText:'IC50 (nM)'}).waitFor();
 await page.getByText('12.5',{exact:true}).waitFor();
 const scroll=await page.evaluate(()=>document.documentElement.scrollWidth);
 if(scroll>w)throw Error(`Results overflow ${w}: ${scroll}`);
 if(process.env.QA_OUTPUT_DIR){fs.mkdirSync(process.env.QA_OUTPUT_DIR,{recursive:true});await page.screenshot({path:`${process.env.QA_OUTPUT_DIR}/results-${w}.png`,fullPage:true});}
 }
 // Load the actual editor assets and check controls at phone, tablet,
 // desktop and short landscape sizes. API data remains isolated fixtures.
 for(const [width,height] of [[320,640],[375,812],[768,700],[1280,720],[1920,1080],[812,375]]){
  await page.setViewportSize({width,height});
  await page.getByRole('button',{name:'Edit structure',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Edit structure'});
  await dialog.waitFor();
  await page.waitForFunction(()=>!document.querySelector('[role=dialog] button.primary')?.disabled);
  const box=await dialog.boundingBox();
  if(box.x<0||box.y<0||box.x+box.width>width+1||box.y+box.height>height+1)throw Error('Editor exceeds viewport '+JSON.stringify({width,height,box}));
  const saveBox=await dialog.getByRole('button',{name:'Save',exact:true}).boundingBox();
  if(!saveBox||saveBox.y+saveBox.height>height)throw Error('Editor Save is off-screen');
  if(width===1280){
   const saveResponse=page.waitForResponse(r=>r.url().endsWith('/structures')&&r.request().method()==='PUT');
   await dialog.getByRole('button',{name:'Save',exact:true}).click();await saveResponse;
   const molblock=submittedRecords?.[0]?.MOLBLOCK;
   if(!molblock||!molblock.split(/\r?\n/)[3]?.includes('V2000'))throw Error('Saved Molfile lost its three-line header');
  }else{await page.keyboard.press('Escape');}
  await dialog.waitFor({state:'hidden'});
 }
 console.log('Editor loaded with visible controls at six viewport sizes');
 recordCount=20;
 await page.setViewportSize({width:375,height:900});
 await page.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&structureTask=structure-qa&assayTask=assay-qa&step=4`);
 await page.locator('.review-card').nth(19).scrollIntoViewIfNeeded();
 await page.locator('.review-card').nth(19).getByAltText('Extracted structure',{exact:true}).waitFor();
 console.log('Card 20 preview rendered after scrolling');
 await page.locator('.review-card').nth(19).getByAltText('Source structure',{exact:true}).waitFor();
 const requestsBeforeLargePage=artifactRequests;
 recordCount=200;
 await page.setViewportSize({width:1280,height:900});
 await page.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&structureTask=structure-qa&assayTask=assay-qa&step=4`);
 await page.locator('.review-card').first().waitFor();
 await page.getByRole('button',{name:'Filters',exact:true}).click();
 await page.getByRole('combobox',{name:'Rows per page',exact:true}).selectOption('200');
 await page.keyboard.press('Escape');
 await page.locator('.review-card').nth(199).scrollIntoViewIfNeeded();
 await page.locator('.review-card').nth(199).getByAltText('Source structure',{exact:true}).waitFor();
 await page.locator('.review-card').first().scrollIntoViewIfNeeded();
 await page.locator('.review-card').first().getByAltText('Source structure',{exact:true}).waitFor();
 const largePageRequests=artifactRequests-requestsBeforeLargePage;
 if(largePageRequests>=80)throw Error(`Off-screen artifacts were eagerly fetched or cache churned: ${largePageRequests}`);
 console.log(`Original-image retry and a 200-row page passed with ${largePageRequests} artifact requests`);
 console.log(JSON.stringify({widths:[320,375,768,1280,1920],errors,assayVisible:true}));
 await browser.close();if(errors.length)process.exit(1);
})().catch(e=>{console.error(e);process.exit(1)});
