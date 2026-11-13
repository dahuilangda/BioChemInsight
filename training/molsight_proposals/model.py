import torch
from torch import nn
from training.molsight_dense.model import DenseConfig,sample_points
from training.molsight_visible.model import VisibleMolsight,VisibleHead,decode_visible


class ProposalHead(VisibleHead):
    def __init__(self,config):
        super().__init__(config)
        self.validity=nn.Linear(config.node_dim,2)
        nn.init.zeros_(self.validity.weight)
        with torch.no_grad():self.validity.bias.copy_(torch.tensor([0.,1.]))

    def proposal_logits(self,encoded,points):
        nodes=self.node(sample_points(encoded['visual'],points).to(encoded['visual'].dtype))
        return self.validity(nodes)


class ProposalMolsight(VisibleMolsight):
    def __init__(self,checkpoint,vocabulary,config=DenseConfig()):
        super().__init__(checkpoint,vocabulary,config);self.head=ProposalHead(config)

    @torch.no_grad()
    def predict_image(self,image):
        self.eval();encoded=self.encode(image)
        try:
            points,scores=self.head.peaks(encoded);raw_count=points.shape[1]
            if not raw_count:raise ValueError('No proposed visible units')
            validity=self.head.proposal_logits(encoded,points).float().softmax(-1)[0,:,1]
            keep=validity>=.5;points=points[:,keep];scores=scores[keep]
            if not points.shape[1]:
                return {'parse_success':False,'smiles':None,'error':'No verified visible units',
                    'predicted_count':0,'raw_candidate_count':raw_count,'points':[],'elements':[],
                    'proposal_validity':validity.cpu().tolist()}
            states=self.head.read(encoded,points,image.shape[-1]);result=decode_visible(points,states)
            result.update(predicted_count=points.shape[1],raw_candidate_count=raw_count,
                points=points[0].float().cpu().tolist(),elements=states['elements'][0].argmax(-1).cpu().tolist(),
                aliases=states['aliases'][0].argmax(-1).cpu().tolist(),atom_scores=scores.cpu().tolist(),
                proposal_validity=validity.cpu().tolist(),proposal_keep=keep.cpu().tolist())
            return result
        except ValueError as error:
            return {'parse_success':False,'smiles':None,'error':str(error),'predicted_count':getattr(error,'count',None)}
