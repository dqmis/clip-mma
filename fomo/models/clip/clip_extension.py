from fomo.models.clip.clip_base import ClipBase
from torch import nn
import torch
from typing_extensions import Self,List, Union
import time

def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class ClipExtension(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        # pass default arguments to the parent class
        super(ClipExtension, self).__init__(backbone, root=root)

        # add additional blocks to the model

        self.image_linear = nn.Linear(512, 37)
        self.text_linear = nn.Linear(37,37)
        
        
        self.image_linear.apply(init_weights)
        self.text_linear.apply(init_weights)
        
        


        
        
    
        

    @property
    def learnable_param_names(self) -> set[str]:
         # IMPORTANT: Add the name of the learnable parameters in the model
        
        return set(["image_linear","text_linear"])

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        self._clip.to(torch.device("cpu"))
        self.image_linear.to(torch.device("cpu"))
        self.text_linear.to(torch.device("cpu"))
        self._clip.float()
        
    def to_cuda(self) -> None:
        self.image_linear.to(torch.device("cuda"))
        self.text_linear.to(torch.device("cuda"))
        self._clip.to(torch.device("cuda"))

        
        

    def forward(self, images: torch.Tensor, prompts: Union[list[str], None] = None) -> torch.Tensor:
        
        #start = time.process_time()
        #print("Start Time:",start)
        # Change the forward method to include the visual_mlp
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images)
        image_features = image_features.to(torch.float32)
        text_features = text_features.to(torch.float32)
        


        
        output1 = self.image_linear(image_features)
        output2 = image_features @ text_features.t()
         
        output3 = output1 + output2
        output4 = self.text_linear(output3)
        
        
        #end = time.process_time()
        #print("Time taken for forward of clip extension:",end-start)
        #logits_per_image: torch.Tensor = self.logit_scale * image_features @ text_features.t()
        #print("Logits shape:",logits_per_image.shape)

        return output4