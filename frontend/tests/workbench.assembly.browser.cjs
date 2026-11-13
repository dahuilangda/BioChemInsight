const {chromium}=require('playwright');
const assert=require('assert/strict');
const {makePdf,thumbnail}=require('./browser-fixtures.cjs');
(async()=>{
 const browser=await chromium.launch({headless:true});
 try{
  for(const available of [true,false]){
   const page=await browser.newPage();let submitted=null;page.on('pageerror',error=>console.error('pageerror',error.message));page.on('console',msg=>{if(msg.type()==='error')console.error(msg.text())});
   await page.route('**/api/**',async route=>{
    const path=new URL(route.request().url()).pathname;if(!path.startsWith('/api/'))return route.continue();let json={};
    if(path.endsWith('/file'))return route.fulfill({contentType:'application/pdf',body:makePdf(2)});
    if(path==='/api/capabilities')json={markush_assembly:{available,requires_review:true}};
    else if(path==='/api/pdfs/qa')json={pdf_id:'qa',filename:'Acceptance.pdf',total_pages:2};
    else if(path.includes('/pages/'))json={image:thumbnail,page:1};
    else if(path==='/api/tasks/full-pipeline'){submitted=route.request().postDataJSON();json={task_id:'assembly-task',type:'full_pipeline',status:'pending',progress:0,params:submitted};}
    else if(path==='/api/tasks')json={tasks:[],total_count:0,total_pages:1,running_count:0,pending_count:0};
    else if(path==='/api/tasks/assembly-task')json={task_id:'assembly-task',type:'full_pipeline',status:'pending',progress:0,params:submitted};
    return route.fulfill({json});
   });
   await page.goto(`${process.env.WORKBENCH_URL||'http://127.0.0.1:5173'}/?pdf=qa&step=1`);
   await page.locator('.step-nav').getByRole('button',{name:'Upload PDF',exact:true}).click();
   const option=page.getByRole('checkbox',{name:'Assemble Markush members',exact:true});await option.waitFor();
   await page.getByText(available?'Optional':'Not available',{exact:true}).waitFor();
   assert.equal(await option.isEnabled(),available);
   if(available){
    await option.check();
    await page.getByRole('button',{name:'Start extraction',exact:true}).click();
    await page.getByRole('button',{name:'Cancel',exact:true}).waitFor();
    assert.equal(submitted.assemble_markush,true);
   }
   assert.equal(await page.getByText('Unavailable',{exact:true}).count(),0);
   await page.close();
  }
  console.log('Markush selection: available control, explicit request flag and unavailable fallback passed');
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
