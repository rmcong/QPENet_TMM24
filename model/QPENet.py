import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        
import model.resnet as models
import model.vgg as vgg_models
from model.cross_transformer import CrossTransformer
from model.ops.modules import MSDeformAttn

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005  
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class Attention(nn.Module):
    def __init__(self, in_channels, drop_rate=0.5):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask=mask
        return mask * embedding

    def forward(self, *x):
        Fs, Ys = x
        att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
        g = self.gate(att)
        Fs = g * Fs
        return Fs

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class GBC(nn.Module):
    def __init__(self, ref_in_planes, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(GBC, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(ref_in_planes, ratio, K, temperature)
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, ref, x):
        softmax_attention = self.attention(ref)
        batch_size, in_planes, height, width = x.size()
        x = x.reshape(1, -1, height, width)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()
        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.pretrained = True
        self.classes = 2

        self.data_set = args.data_set
        self.split = args.split
        ppm_scales=[60, 30, 15, 8]
        
        self.map_mode = 'Cosine'
        assert self.layers in [50, 101, 152]

        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            print('INFO: Using ResNet {}'.format(self.layers))
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            else:
                resnet = models.resnet152(pretrained=self.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        self.transformer = CrossTransformer(embed_dims=reduce_dim, shot=self.shot, num_points=9)

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )


        mask_add_num = 1
        self.init_merge1 = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + mask_add_num*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.init_merge2 = nn.Sequential(
            nn.Conv2d(reduce_dim*3 + mask_add_num*4, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))           
        self.fg_res = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                          
        )                         
        self.fg_cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )
        self.merge_multi_lvl_reduce = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                )
        self.merge_multi_lvl_sum = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))
        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        self.down_bg = nn.Sequential(
            nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        ) 
        self.bg_res1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        ) 
        self.bg_cls = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.1),                 
                    nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )
        self.bg_prototype = nn.Parameter(torch.zeros(1, reduce_dim,1,1))
        self.bg_loss = nn.CrossEntropyLoss(reduction='none')
        self.gam=Attention(in_channels=256)
        self.Dynamic_conv = GBC(reduce_dim, reduce_dim, reduce_dim, 3, padding=1, bias=False)
        self.down_cyc = nn.Sequential(
            nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        ) 
        self.cyc_res1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )
        self.cyc_cls = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.1),                 
                    nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )



    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
                {'params': model.bg_prototype},
                {'params': model.down_bg.parameters()},
                {'params': model.bg_res1.parameters()},
                {'params': model.bg_cls.parameters()},
                {'params': model.gam.parameters()},
                {'params': model.Dynamic_conv.parameters()},
                {'params': model.down_cyc.parameters()},
                {'params': model.cyc_res1.parameters()},
                {'params': model.cyc_cls.parameters()},
                {'params': model.transformer.parameters()},
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge1.parameters()},
                {'params': model.init_merge2.parameters()},
                {'params': model.fg_res.parameters()},
                {'params': model.fg_cls.parameters()},
                {'params': model.merge_multi_lvl_reduce.parameters()},
                {'params': model.merge_multi_lvl_sum.parameters()},
                {'params': model.cls_meta.parameters()}
            ],
            lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)  # 2.5e-3, 0.9, 1e-4
        
        return optimizer


    def forward(self, s_x, s_y,x, y, classes, padding_mask=None, s_padding_mask=None, prototype_neg_dict=None, method_weight=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)  
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        supp_feat_list = []
        supp_nomask_feat_list = []
        supp_simple_out_list = []
        final_supp_list = []
        mask_list = []
        bg_mask_list = []
        supp_feat_alpha_list = []
        supp_feat_gamma_list = []
        supp_feat_delta_list = []
        supp_feat_beta_list = []
        supp_feat_BG_list = []
        gams = 0
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            bg_mask = (s_y[:,i,:,:] == 0)
            mask_list.append(mask)
            bg_mask_list.append(bg_mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                supp_feat_4_true = self.layer4(supp_feat_3)

                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                mask_1 = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='nearest')
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat_nomask = self.down_supp(supp_feat)
            supp_feat = Weighted_GAP(supp_feat_nomask, mask)

            gams += self.gam(supp_feat_nomask, mask)
            supp_feat_cyc = supp_feat.expand(-1, -1, query_feat.size(2), query_feat.size(3))
            qry_cyc_feat = torch.cat((query_feat,supp_feat_cyc),dim=1)
            qry_cyc_feat_1 = self.down_cyc(qry_cyc_feat)
            qry_cyc_feat_2 = self.cyc_res1(qry_cyc_feat_1) + qry_cyc_feat_1     
            query_simple_out = self.cyc_cls(qry_cyc_feat_2)
            query_mask_simple = F.interpolate(query_simple_out, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            query_mask_pre = query_mask_simple.max(1)[1].unsqueeze(1).float()  # 60*60
            query_fear = Weighted_GAP(query_feat, query_mask_pre)
            query_fear_cyc = query_fear.expand(-1, -1, query_feat.size(2), query_feat.size(3))
            supp_cyc_feat = torch.cat((supp_feat_nomask,query_fear_cyc),dim=1)
            supp_cyc_feat_1 = self.down_cyc(supp_cyc_feat)
            supp_cyc_feat_2 = self.cyc_res1(supp_cyc_feat_1) + supp_cyc_feat_1 
            supp_simple_out = self.cyc_cls(supp_cyc_feat_2)

            supp_simple_out_list.append(supp_simple_out)  #simple
            mask_simple = F.interpolate(supp_simple_out, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            mask_simple_pre = mask_simple.max(1)[1].unsqueeze(1)  # 60*60
            mask_alpha = mask_simple_pre * mask_1
            mask_delta = mask_simple_pre - mask_alpha
            mask_beta = mask_1 - mask_alpha
            mask_gamma = 1- mask_alpha - mask_delta - mask_beta
            mask_BG = 1 - mask_1
            
            supp_feat_alpha = Weighted_GAP(supp_feat_nomask, mask_alpha)
            supp_feat_delta = Weighted_GAP(supp_feat_nomask, mask_delta)
            supp_feat_gamma = Weighted_GAP(supp_feat_nomask, mask_gamma)
            supp_feat_beta = Weighted_GAP(supp_feat_nomask, mask_beta)

            supp_feat_alpha_list.append(supp_feat_alpha)
            supp_feat_gamma_list.append(supp_feat_gamma)
            supp_feat_delta_list.append(supp_feat_delta)
            supp_feat_beta_list.append(supp_feat_beta)

            supp_feat_list.append(supp_feat)
            supp_nomask_feat_list.append(supp_feat_nomask)

        gam = gams / self.shot
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask                    
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)     
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)  

        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list)

            supp_feat_nomask = supp_nomask_feat_list[0]
            for i in range(1, len(supp_nomask_feat_list)):
                supp_feat_nomask += supp_nomask_feat_list[i]
            supp_feat_nomask /= len(supp_nomask_feat_list)

            supp_feat_alpha = supp_feat_alpha_list[0]
            for i in range(1, self.shot):
                supp_feat_alpha += supp_feat_alpha_list[i]
            supp_feat_alpha /= self.shot

            supp_feat_gamma = supp_feat_gamma_list[0]
            for i in range(1, self.shot):
                supp_feat_gamma += supp_feat_gamma_list[i]
            supp_feat_gamma /=self.shot

            supp_feat_beta = supp_feat_beta_list[0]
            for i in range(1, self.shot):
                supp_feat_beta += supp_feat_beta_list[i]
            supp_feat_beta /=self.shot

            supp_feat_delta = supp_feat_delta_list[0]
            for i in range(1, self.shot):
                supp_feat_delta += supp_feat_delta_list[i]
            supp_feat_delta /=self.shot

        pro_alpha = supp_feat_alpha
        pro_beta = supp_feat_beta
        pro_gamma = supp_feat_gamma
        pro_delta = supp_feat_delta
        bg = self.bg_prototype.expand(query_feat.size(0),-1,query_feat.size(2),query_feat.size(3))

        pro_map = torch.cat([self.bg_prototype.expand(query_feat.size(0),-1,1,1).unsqueeze(1) , pro_alpha.unsqueeze(1) , \
                supp_feat.unsqueeze(1) , pro_beta.unsqueeze(1) , pro_delta.unsqueeze(1), pro_gamma.unsqueeze(1)], 1)
        activation_map = self.query_region_activate(query_feat, pro_map , self.map_mode).unsqueeze(2) #b,6,1,h,w

        qrybg_feat = torch.cat((query_feat,bg),dim=1)
        qrybg_feat1 = self.down_bg(qrybg_feat)
        qrybg_feat2 = self.bg_res1(qrybg_feat1) + qrybg_feat1         
        query_bg_out = self.bg_cls(qrybg_feat2)
        query_bg_mask = query_bg_out.max(1)[1]
        query_feat_withmask = query_feat * ((query_bg_mask == 1).float().unsqueeze(1))
        query_bg_prototype = self.Dynamic_conv(query_feat_withmask, bg)
        
        supp_bg_out_list = []
        if self.training:
            for supp_feat_nomask in supp_nomask_feat_list:
                suppbg_feat = torch.cat((supp_feat_nomask,bg),dim=1)
                suppbg_feat = self.down_bg(suppbg_feat)
                suppbg_feat = self.bg_res1(suppbg_feat) + suppbg_feat          
                supp_bg_out = self.bg_cls(suppbg_feat)
                supp_bg_out_list.append(supp_bg_out)

        inital_out_list = []

        supp_feat = supp_feat * 0.5 + query_fear * 0.5
        supp_feat_bin = supp_feat.expand_as(query_feat)
        bg_feat_bin = query_bg_prototype
        corr_mask_bin = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
        activation_1 = F.interpolate(activation_map[:,1,...], size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
        activation_2 = F.interpolate(activation_map[:,2,...], size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
        activation_3 = F.interpolate(activation_map[:,3,...], size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
        activation_4 = F.interpolate(activation_map[:,4,...], size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
        activation_5 = F.interpolate(activation_map[:,5,...], size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True) 
        merge_feat_binbg = torch.cat([query_feat, bg_feat_bin, activation_4, activation_5],1)
        merge_feat_binbg = self.init_merge1(merge_feat_binbg)
        merge_feat_binfg = torch.cat([merge_feat_binbg, supp_feat_bin, corr_mask_bin, activation_1, activation_2, activation_3, gam],1)
        merge_feat_binfg = self.init_merge2(merge_feat_binfg)
        merge_feat_binfg = self.fg_res(merge_feat_binfg) + merge_feat_binfg   
        inital_out = self.fg_cls(merge_feat_binfg)
        inital_out_list.append(inital_out)
        query_feat_list = self.transformer(merge_feat_binfg, padding_mask.float(), gam, s_y.clone().float(), s_padding_mask.float())
        fused_query_feat = []
        for lvl, qry_feat in enumerate(query_feat_list):
            if lvl == 0:
                fused_query_feat.append(qry_feat)
            else:
                fused_query_feat.append(F.interpolate(qry_feat, size=(query_feat.shape[-2:][0], query_feat.shape[-2:][1]), mode='bilinear', align_corners=True))
        merge_feat_bin_attention = torch.cat(fused_query_feat, dim=1)
        fused_query_feat = self.merge_multi_lvl_reduce(merge_feat_bin_attention)
        fused_query_feat = self.merge_multi_lvl_sum(fused_query_feat)+fused_query_feat 
        out = self.cls_meta(fused_query_feat)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            query_simple_out = F.interpolate(query_simple_out, size=(h, w), mode='bilinear', align_corners=True)
            for i in range(self.shot):
                supp_simple_out_list[i]  = F.interpolate(supp_simple_out_list[i] , size=(h, w), mode='bilinear', align_corners=True)
                mask_list[i] = F.interpolate(mask_list[i], size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            query_bg_out = F.interpolate(query_bg_out, size=(h, w), mode='bilinear', align_corners=True)
            supp_bg_out = F.interpolate(supp_bg_out, size=(h, w), mode='bilinear', align_corners=True)
            erfa = 0.5
            main_loss = self.criterion(out, y.long())
            aux_loss2 = torch.zeros_like(main_loss).cuda()    

            for idx in range(len(inital_out_list)):    
                inner_out = inital_out_list[idx]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss2 = aux_loss2 + self.criterion(inner_out, y.long())   
            aux_loss2 = aux_loss2 / len(inital_out_list)

            mygt1 = torch.ones(query_bg_out.size(0),h,w).cuda()
            mygt0 = torch.zeros(query_bg_out.size(0),h,w).cuda()
            query_bg_loss = self.weighted_BCE(query_bg_out, mygt0, y)+erfa*self.criterion(query_bg_out,mygt1.long())
            aux_bg_loss = 0.
            for j,supp_bg_out in enumerate(supp_bg_out_list):
                supp_bg_out = F.interpolate(supp_bg_out, size=(h, w), mode='bilinear', align_corners=True)
                supp_bg_loss = self.weighted_BCE(supp_bg_out, mygt0, s_y[:,j,:,:])+erfa*self.criterion(supp_bg_out,mygt1.long())
                aux_bg_loss = aux_bg_loss + supp_bg_loss

            bg_loss = (query_bg_loss + aux_bg_loss)/ (len(supp_bg_out_list)+1)     

            query_aux_loss = self.criterion(query_simple_out, y.long())  
            for i in range(self.shot):
                query_aux_loss += self.criterion(supp_simple_out_list[i], mask_list[i].squeeze(1).long())
            query_aux_loss = query_aux_loss/(1+self.shot)

            return out.max(1)[1], main_loss,bg_loss+0.6*aux_loss2+0.5*query_aux_loss,prototype_neg_dict
        else:
            return out

    def weighted_BCE(self,input, target,mask):
        loss_list =[]
        cmask = torch.where(mask.long() == 1,mask.long(),target.long())
        
        for x,y,z in zip(input,target,cmask):
            loss = self.bg_loss(x.unsqueeze(0),y.unsqueeze(0).long())
            area = torch.sum(z)+1e-5
            Loss = torch.sum(z.unsqueeze(0)*loss) /area
            loss_list.append(Loss.unsqueeze(0))
        LOSS = torch.cat(loss_list,dim=0)                     
        return torch.mean(LOSS)

    def query_region_activate(self, query_fea, prototypes, mode):
        b, c, h, w = query_fea.shape
        n = prototypes.shape[1]
        que_temp = query_fea.reshape(b, c, h*w)

        if mode == 'Conv':
            map_temp = torch.bmm(prototypes.squeeze(-1).squeeze(-1), que_temp)  # [b, n, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            return activation_map

        elif mode == 'Cosine':
            que_temp = que_temp.unsqueeze(dim=1)           # [b, 1, c, h*w]
            prototypes_temp = prototypes.squeeze(dim=-1)   # [b, n, c, 1]
            map_temp = nn.CosineSimilarity(2)(que_temp, prototypes_temp)  # [n, c, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            return activation_map

        elif mode == 'Learnable':
            for p_id in range(n):
                prototypes_temp = prototypes[:,p_id,:,:,:]                         # [b, c, 1, 1]
                prototypes_temp = prototypes_temp.expand(b, c, h, w)
                concat_fea = torch.cat([query_fea, prototypes_temp], dim=1)        # [b, 2c, h, w]                
                if p_id == 0:
                    activation_map = self.relation_coding(concat_fea)              # [b, 1, h, w]
                else:
                    activation_map_temp = self.relation_coding(concat_fea)              # [b, 1, h, w]
                    activation_map = torch.cat([activation_map,activation_map_temp], dim=1)
            return activation_map
            