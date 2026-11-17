// Seeded geometry invariants across 20,000 gestures, without a browser/model.
const ts=require('../ui/node_modules/typescript');
const {Module}=require('module');const fs=require('fs');const assert=require('assert/strict');
const file=require('path').resolve(__dirname,'../ui/src/utils/pdfRegion.ts');const m=new Module(file);m._compile(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText,file);
const {adjustRegion}=m.exports;let seed=713;
const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
for(let i=0;i<20000;i++){
 const start={x:random(),y:random()},end={x:random(),y:random()};const box=adjustRegion('new',start,end);
 assert.ok(Math.abs(box.x+box.w-Math.max(start.x,end.x))<1e-12);
 for(const action of ['move','nw','ne','sw','se']) {
  const b=adjustRegion(action,start,{x:random(),y:random()},box);
  assert.ok([b.x,b.y,b.w,b.h].every(Number.isFinite));
  assert.ok(b.x>=0&&b.y>=0&&b.w>=0&&b.h>=0&&b.x+b.w<=1+1e-12&&b.y+b.h<=1+1e-12);
  if(action==='move'){assert.equal(b.w,box.w);assert.equal(b.h,box.h);}
 }
}
console.log(JSON.stringify({gestures:20000,transformations:120000,bounds:true,movePreservesSize:true,reverseDrawing:true}));
