"""Real-image GPU BF16 stress cases with zero-learning-rate optimizer steps."""
import argparse,json
from pathlib import Path
import pyarrow.compute as pc
import torch
from types import SimpleNamespace
from molsight.checkpoint import load_parameters,model_state
from molsight.model import MolsightModel
from molsight.prepared import PreparedDataset,collate_characters,file_hash
from molsight.retention import RetentionObjective
from molsight.tokenizer import CharTokenizer
from molsight_multitask.data import GraphDataset,collate_graphs
from molsight_multitask.objective import GraphCriterion
from molsight_multitask.replay import DomainReplaySampler
from molsight_multitask.train import create_student,validate_configuration,parameter_groups
from molsight_multitask.update import multitask_update


def run(experiment,output,device):
 config=json.loads(Path(experiment).read_text());validate_configuration(config)
 data=GraphDataset(config['manifest'],'train',sidecar=config['train_sidecar']);data._open();table=data.table
 tokenizer=CharTokenizer(data.specification['vocabulary'])
 initial=torch.load(config['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
 torch.manual_seed(config['seed'])
 student,configuration,new=create_student(initial,tokenizer)
 snapshot={key:value.clone() for key,value in student.state_dict().items()}
 student.to(device)
 reference=MolsightModel(SimpleNamespace(**initial['plan']['configuration']),tokenizer)
 load_parameters(reference,model_state(initial),source_vocabulary=tokenizer.stoi,target_vocabulary=tokenizer.stoi)
 reference.requires_grad_(False).eval().to(device)
 retention=RetentionObjective(reference,weight=config['retention_weight'],supervised_weight=config['retention_supervised_weight'],temperature=config['retention_temperature'])
 anchor_data=PreparedDataset(config['manifest'],'train')
 anchors=[]
 for domain,weight in sorted(config['retention_domains'].items()):
  sampler=DomainReplaySampler(anchor_data,domain,vocabulary_size=len(tokenizer),updates=2,batch_size=1,seed=config['seed'])
  anchors.append({'name':domain,'weight':weight,'batch':collate_characters([anchor_data[next(iter(sampler))[0]]])})
 lengths=pc.list_value_length(table['token_ids'])
 peptide_indices=pc.indices_nonzero(pc.equal(table['category'],'peptide')).to_pylist()
 high_indices=pc.indices_nonzero(pc.equal(table['input_size'],1024)).to_pylist()
 cases={'longest_peptide':max(peptide_indices,key=lambda index:lengths[index].as_py()),
        'longest_1024_input':max(high_indices,key=lambda index:lengths[index].as_py())}
 results={}
 criterion=GraphCriterion(SimpleNamespace(**configuration),tokenizer,edge_weight=config['edge_weight']).to(device)
 for name,index in cases.items():
  student.train();student.image_encoder.eval()
  optimizer=torch.optim.AdamW(parameter_groups(student,config,True),weight_decay=config['weight_decay'])
  scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
  batch=collate_graphs([data[index]])
  torch.cuda.reset_peak_memory_stats(device)
  losses,norm,stepped=multitask_update(student,criterion,[batch],optimizer,scheduler,torch.amp.GradScaler('cuda',enabled=False),device,config['max_grad_norm'],amp=True,retention=retention,anchor_batch=anchors)
  if not stepped or any(not torch.equal(value,student.state_dict()[key].detach().cpu()) for key,value in snapshot.items()):
   raise RuntimeError('Zero-LR stress update changed state or did not commit')
  results[name]={'record_index':index,'record_key':table['record_key'][index].as_py(),'tokens':int(lengths[index].as_py()),'atoms':int(batch['atom_indices'].shape[1]),'input_size':int(batch['image'].shape[-1]),'losses':losses,'gradient_norm':float(norm),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'original_state_unchanged':True}
  del optimizer,scheduler
  torch.cuda.empty_cache()
 report={'passed':True,'device':str(device),'precision':'bf16','experiment_sha256':file_hash(experiment),'cases':results,'scope':'Single-GPU real-image stress updates plus multidomain replay. Complements two-rank BF16 and CPU global-gradient tests; not a capability result or exhaustive memory bound.'}
 with Path(output).open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',type=torch.device,default=torch.device('cuda:0'));args=parser.parse_args();run(args.experiment,args.out,args.device)
