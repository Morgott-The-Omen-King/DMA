#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset

import os
import json
import random
from typing import List, Dict

class ActionHierarchy:
    def __init__(self, action_groups_path: str, group_id, train=False):
        """初始化动作层次结构分析器
        
        Args:
            action_groups_path: action_groups.json的路径
        """
        self.action_groups_path = action_groups_path
        self.action_groups = self._load_action_groups()
        if train:
            self.action_groups = [self.action_groups[i] for i in range(4) if i != group_id]
        else:
            self.action_groups = [self.action_groups[group_id]]
        self.action_hierarchy = self._build_hierarchy()

        
    def _load_action_groups(self) -> List[Dict]:
        """加载action_groups.json文件"""
        with open(self.action_groups_path, 'r') as f:
            return json.load(f)
    
    def _build_hierarchy(self) -> Dict:
        """构建动作类别的层次结构"""
        hierarchy = {}
        
        for group in self.action_groups:
            for action in group['actions']:
                parts = action.split('-')
                if len(parts) >= 2:
                    current_level = hierarchy
                    for i, part in enumerate(parts[:-1]):
                        if part not in current_level:
                            current_level[part] = {}
                        current_level = current_level[part]
                    if parts[-1]:  # 如果最后一级不为空
                        if 'actions' not in current_level:
                            current_level['actions'] = set()
                        current_level['actions'].add(parts[-1])
        
        return hierarchy
    
    def get_similar_actions(self, action: str, num_actions: int, level: int = None) -> List[str]:
        """获取与给定动作相似的动作列表
        
        Args:
            action: 目标动作
            num_actions: 需要返回的相似动作数量
            level: 指定搜索的层级,None表示使用原始层级
            
        Returns:
            相似动作列表
        """
        parts = action.split('-')
        if level is not None:
            parts = parts[:level]
        similar_actions = []
        
        def traverse_hierarchy(current_dict, current_path):
            if 'actions' in current_dict:
                for act in current_dict['actions']:
                    full_path = '-'.join(current_path + [act])
                    if full_path != action:
                        similar_actions.append(full_path)
            for key in current_dict:
                if key != 'actions' and isinstance(current_dict[key], dict):
                    traverse_hierarchy(current_dict[key], current_path + [key])
        
        # 从相同前缀开始搜索
        current_dict = self.action_hierarchy
        for i in range(len(parts)-1):
            if parts[i] in current_dict:
                current_dict = current_dict[parts[i]]
            else:
                break
        
        traverse_hierarchy(current_dict, parts[:-1])
        
        if len(similar_actions) > num_actions:
            return random.sample(similar_actions, num_actions)
        return similar_actions
    
    def sample_fine_grained_episode(self, num_ways: int) -> List[str]:
        """采样细粒度的episode
        
        Args:
            num_ways: N-way中的N
            
        Returns:
            选中的动作类别列表
        """
        all_actions = []
        
        def collect_actions(current_dict, current_path):
            if 'actions' in current_dict:
                for act in current_dict['actions']:
                    all_actions.append('-'.join(current_path + [act]))
            for key in current_dict:
                if key != 'actions' and isinstance(current_dict[key], dict):
                    collect_actions(current_dict[key], current_path + [key])
        
        collect_actions(self.action_hierarchy, [])
        
        base_action = random.choice(all_actions)
        selected_actions = [base_action]
        
        # 从最细粒度开始尝试
        current_level = len(base_action.split('-'))
        while len(selected_actions) < num_ways and current_level > 1:
            similar_actions = self.get_similar_actions(base_action, num_ways - len(selected_actions), current_level)
            selected_actions.extend(similar_actions)
            
            # 如果当前层级采样不够,向上走一层
            if len(selected_actions) < num_ways:
                current_level -= 1
                
        # 如果还是不够N-way,从其他节点随机采样补充
        if len(selected_actions) < num_ways:
            remaining_actions = list(set(all_actions) - set(selected_actions))
            if remaining_actions:
                additional_actions = random.sample(remaining_actions, min(num_ways - len(selected_actions), len(remaining_actions)))
                selected_actions.extend(additional_actions)
        
        return selected_actions[:num_ways]


class MoVeDataset(Dataset):
    def __init__(self, 
                 train=True, 
                 valid=False,
                 finetune_idx=None,
                 support_frames=10, 
                 query_frames=1,  # N-way-K-shot setting
                 num_ways=1, 
                 num_shots=5,  # N-way-K-shot setting
                 transforms=None, 
                 another_transform=None, 
                 group=0, 
                 setting='default',
                 sample_method='hard'):
        self.train = train
        self.valid = valid
        
        self.sample_method = sample_method
        assert self.sample_method in ['hard', 'random'], 'sample_method must be either "hard" or "random"'
        
        self.support_frames = support_frames
        self.query_frames = query_frames
        self.num_ways = num_ways  # N-way
        self.num_shots = num_shots  # K-shot
        
        self.transforms = transforms
        self.another_transform = another_transform
        self.group = group
        self.setting = setting
        
        # Setup data paths
        self.data_dir = os.path.join('data', 'MoVe')
        self.img_dir = os.path.join(self.data_dir, 'frames')
        self.ann_dir = os.path.join(self.data_dir, 'annotations')

        # Initialize data structures
        self.video_ids = []
        self.action_categories = set()  # Store unique action categories
        self.category_to_videos = {}  # Map categories to video IDs
        self.video_to_categories = {}  # Map videos to their action categories
        self.action_segments = {}  # Store video length and objects info
        
        # Load annotations and collect action categories
        for vid in os.listdir(self.img_dir):
            if not os.path.isdir(os.path.join(self.img_dir, vid)):
                continue
                
            ann_file = os.path.join(self.ann_dir, f"{vid}.json")
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
                    if 'length' in ann_data and 'objects' in ann_data:
                        # Store basic video info
                        self.action_segments[vid] = {
                            'length': ann_data['length'],
                            'objects': ann_data['objects']
                        }
                        
                        # Extract action categories
                        video_categories = set()
                        for obj in ann_data['objects']:
                            if 'actions' in obj:
                                for action_info in obj['actions']:
                                    action_name = action_info['action']
                                    self.action_categories.add(action_name)
                                    video_categories.add(action_name)
                                    
                                    # Add to category mapping
                                    if action_name not in self.category_to_videos:
                                        self.category_to_videos[action_name] = []
                                    if vid not in self.category_to_videos[action_name]:
                                        self.category_to_videos[action_name].append(vid)
                        
                        # Store video's categories
                        self.video_to_categories[vid] = list(video_categories)

        # Sort categories for reproducibility
        self.action_categories = sorted(list(self.action_categories))
        
        # Load action groups from JSON
        if setting == 'default':
            action_groups_path = os.path.join(os.path.dirname(self.ann_dir), 'action_groups.json')
        elif setting == 'challenging':
            action_groups_path = os.path.join(os.path.dirname(self.ann_dir), 'challenging_group.json')
        else: assert False, 'setting input is not valid! please try "default" or "challenging"'
        with open(action_groups_path, 'r') as f:
            action_groups = json.load(f)
        
        # 只选择两个类别
        test_categories = action_groups[self.group]['actions'][:2]  # 只取前两个类别
        train_categories = [cat for cat in self.action_categories if cat not in test_categories]
        self.train_categories = train_categories
        self.test_categories = test_categories
        
        if train and not valid:
            self.selected_categories = train_categories
        else:
            self.selected_categories = test_categories
        
        self.selected_categories = ["daily_action-sign_language-cool"]
            
        # 限制每个类别的视频数量
        self.video_ids = []
        for i, category in enumerate(self.selected_categories):
            videos = self.category_to_videos[category]
            if i == 0:  # 第一个类别只选1个视频
                self.video_ids.extend(videos[:2])
            else:  # 第二个类别选2个视频
                self.video_ids.extend(videos[:2])
        
        if finetune_idx is not None:
            self.video_ids = [self.video_ids[finetune_idx]]

        print(f"{'Train' if train else 'Test'} set: {len(self.video_ids)} videos")
        print(f"Number of action categories: {len(self.selected_categories)}")
        if len(self.selected_categories) < self.num_ways:
            raise ValueError(f"Not enough categories for {self.num_ways}-way setting")

        if train and not valid:
            self.action_hierarchy = ActionHierarchy(action_groups_path, self.group, train=True)
        else:
            self.action_hierarchy = ActionHierarchy(action_groups_path, self.group, train=False)

    def get_frames(self, video_id, frame_indices, action_name=None):
        """
        Get frames and corresponding masks for a specific action
        Args:
            video_id: ID of the video
            frame_indices: List of frame indices to get
            action_name: Target action name to get masks for
        """
        frames = []
        masks = []
        video_dir = os.path.join(self.img_dir, video_id)
        
        # Get all frame files and sort them
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        
        # Find objects that have the target action
        target_objects = []
        if action_name is not None and video_id in self.action_segments:
            for obj in self.action_segments[video_id]['objects']:
                if 'actions' in obj:
                    for action_info in obj['actions']:
                        if action_info['action'] == action_name:
                            # Check if frame is within action's time range
                            start_frame = action_info.get('start_frame', 0)
                            end_frame = action_info.get('end_frame', len(frame_files) - 1)
                            target_objects.append({
                                'object': obj,
                                'start_frame': start_frame,
                                'end_frame': end_frame
                            })
        
        # Load selected frames
        for idx in frame_indices:
            # Load image
            try:
                img_path = os.path.join(video_dir, frame_files[idx])
            except:
                print(f"Error loading image at index {idx} for video {video_id}")
            img = np.array(Image.open(img_path))
            frames.append(img)
            
            # Create combined mask for all objects with target action at this frame
            h, w = img.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            if action_name is not None and target_objects:
                for target in target_objects:
                    obj = target['object']
                    start_frame = target['start_frame']
                    end_frame = target['end_frame']
                    
                    # Only include mask if frame is within action's time range
                    if start_frame <= idx <= end_frame:
                        if obj['masks'][idx] is not None:
                            obj_mask = self.decode_mask(obj['masks'][idx])[:, :, 0]
                            combined_mask = np.logical_or(combined_mask, obj_mask)
            
            masks.append(combined_mask)
        
        return frames, masks

    def decode_mask(self, mask_data):
        from pycocotools import mask as mask_utils
        
        h, w = mask_data['size']
        rle = {'size': [h, w], 'counts': mask_data['counts']}
        
        # Use COCO's decode function
        mask = mask_utils.decode(rle)
        
        # Add channel dimension and ensure uint8 type
        mask = np.expand_dims(mask, axis=2).astype(np.uint8)
        
        return mask

    def get_valid_frames(self, video_id, action_name):
        """获取视频中包含指定动作的帧范围信息
        Returns:
            action_frames: 包含动作的帧列表
            all_frames: 视频中所有可用帧的列表
        """
        action_frames = []
        video_data = self.action_segments[video_id]
        video_length = video_data['length']
        all_frames = list(range(video_length))
        
        for obj in video_data['objects']:
            if 'actions' in obj:
                for action_info in obj['actions']:
                    if action_info['action'] == action_name:
                        start_frame = action_info.get('start_frame', 0)
                        end_frame = action_info.get('end_frame', video_length - 1)
                        end_frame = min(end_frame, video_length - 1)
                        action_frames.extend(range(start_frame, end_frame + 1))
        
        return sorted(list(set(action_frames))), all_frames

    def sample_frames_with_action(self, valid_frames, all_frames, num_frames, min_action_frames=1):
        """采样帧，确保至少包含指定数量的动作帧
        Args:
            valid_frames: 包含目标动作的帧列表
            all_frames: 所有可用帧列表
            num_frames: 需要采样的总帧数
            min_action_frames: 最少需要包含的动作帧数量
        Returns:
            采样的帧索引列表
        """
        if not valid_frames:
            return random.sample(all_frames, num_frames)
            
        # 确保至少采样min_action_frames个动作帧
        num_action_frames = min(len(valid_frames), min(min_action_frames, num_frames))
        selected_action_frames = random.sample(valid_frames, num_action_frames)
        
        # 剩余帧数从所有帧中随机采样
        remaining_frames = num_frames - num_action_frames
        other_frames = [f for f in all_frames if f not in selected_action_frames]
        if remaining_frames > 0:
            if len(other_frames) >= remaining_frames:
                selected_other_frames = random.sample(other_frames, remaining_frames)
            else:
                try:
                    selected_other_frames = random.choices(other_frames, k=remaining_frames)
                except:
                    print(f"Error sampling other frames for video {video_id}")
                    selected_other_frames = []
        else:
            selected_other_frames = []
        
        # 合并所有选择的帧并按时间顺序排序
        selected_frames = sorted(selected_action_frames + selected_other_frames)
        return selected_frames
    
    def sample_frames_with_action_support(self, valid_frames, all_frames, num_frames):
        """采样支持帧，确保采样完整的动作序列
        Args:
            valid_frames: 包含目标动作的帧列表
            all_frames: 所有可用帧列表 
            num_frames: 需要采样的总帧数
        Returns:
            采样的帧索引列表
        """
        if not valid_frames:
            # 如果没有有效动作帧，则等间距采样
            if len(all_frames) <= num_frames:
                return sorted(all_frames)
            step = len(all_frames) // num_frames
            return sorted(all_frames[::step][:num_frames])
            
        # 找到连续的动作片段
        action_segments = []
        current_segment = []
        for i in range(len(valid_frames)):
            if not current_segment or valid_frames[i] == current_segment[-1] + 1:
                current_segment.append(valid_frames[i])
            else:
                action_segments.append(current_segment)
                current_segment = [valid_frames[i]]
        if current_segment:
            action_segments.append(current_segment)
            
        # 选择最长的动作片段
        longest_segment = max(action_segments, key=len)
        
        if len(longest_segment) >= num_frames:
            # 如果最长片段长度大于需要的帧数，等间距采样
            step = len(longest_segment) // num_frames
            return sorted(longest_segment[::step][:num_frames])
        else:
            # 如果最长片段长度小于需要的帧数，向两边扩展采样
            center = longest_segment[len(longest_segment)//2]
            left = center - 1
            right = center + 1
            selected_frames = longest_segment.copy()
            
            while len(selected_frames) < num_frames and (left >= 0 or right < len(all_frames)):
                if left >= 0 and len(selected_frames) < num_frames:
                    selected_frames.append(left)
                    left -= 1
                if right < len(all_frames) and len(selected_frames) < num_frames:
                    selected_frames.append(right)
                    right += 1
                    
            return sorted(selected_frames[:num_frames])
    
    def __getitem__(self, idx):
        if self.train:
            return self.__gettrainitem__(idx)
        else:
            return self.__gettestitem__(idx)

    def __gettrainitem__(self, idx):
        # selected_categories = random.sample(self.selected_categories, self.num_ways)
        # Prepare support and query data
        all_support_frames = []
        all_support_masks = []
        all_query_frames = []
        all_query_masks = []

        # if random.random() < 0.7:
        if False:
            selected_categories = self.action_hierarchy.sample_fine_grained_episode(self.num_ways)
            query_category = random.choice(selected_categories)
            selected_categories = selected_categories[:self.num_ways]

            # 获取该类别的所有视频
            category_videos = self.category_to_videos[query_category]
            
            # 确保有足够的视频用于support和query
            if len(category_videos) < self.num_shots + 1:
                # 如果视频不够，允许重复使用
                # support_video_ids = random.choices([v for v in category_videos if v != query_category], k=self.num_shots)
                query_video_id = random.choice(category_videos)
            else:
                # 随机选择support视频
                # support_video_ids = random.sample([v for v in category_videos if v != query_category], self.num_shots)
                # 从剩余视频中选择query视频
                # remaining_videos = [v for v in category_videos if v not in support_video_ids]
                query_video_id = random.choice(category_videos)
        else:
            selected_categories = self.selected_categories
            query_category = selected_categories[0]
            selected_categories = selected_categories[:self.num_ways]
        
            # Find videos that contain at least one of the selected categories
            eligible_query_videos = []
            for vid in self.video_ids:
                if any(cat in self.video_to_categories[vid] for cat in selected_categories):
                    eligible_query_videos.append(vid)
            
            if len(eligible_query_videos) < 1:
                raise ValueError("Not enough eligible query videos")
            
            # # Randomly select one query video
            # query_video_id = random.choice(eligible_query_videos)
            
            # 获取该类别的所有视频
            category_videos = self.category_to_videos[query_category]
            
            # 确保有足够的视频用于support和query
            if len(category_videos) < self.num_shots + 1:
                # 如果视频不够，允许重复使用
                # support_video_ids = random.choices([v for v in category_videos if v != query_category], k=self.num_shots)
                query_video_id = random.choice(category_videos)
            else:
                # 随机选择support视频
                # support_video_ids = random.sample([v for v in category_videos if v != query_category], self.num_shots)
                # 从剩余视频中选择query视频
                # remaining_videos = [v for v in category_videos if v not in support_video_ids]
                query_video_id = random.choice(category_videos)
        query_video_id = 'b2b1orzu9hex'
        valid_query_frames, all_query_frames_list = self.get_valid_frames(query_video_id, query_category)
        print(query_video_id, query_category)
        
        # 为query帧确保至少包含1个动作帧
        query_indices = self.sample_frames_with_action(
            valid_query_frames,
            all_query_frames_list,
            self.query_frames,
            min_action_frames=1
        )
        
        query_frames, query_masks = self.get_frames(query_video_id, query_indices, query_category)
        
        # For each category, get support data and corresponding query masks
        for category in selected_categories:
            # Get support videos (excluding query video)
            available_support_videos = [v for v in self.category_to_videos[category] 
                                     if v != query_video_id]
            
            if len(available_support_videos) < self.num_shots:
                support_video_ids = random.choices(available_support_videos, k=self.num_shots)
            else:
                support_video_ids = random.sample(available_support_videos, self.num_shots)
            
            # Get support data
            for support_video_id in support_video_ids:
                valid_support_frames, all_support_frames_list = self.get_valid_frames(support_video_id, category)
                
                shot_indices = self.sample_frames_with_action_support(
                    valid_support_frames,
                    all_support_frames_list,
                    self.support_frames
                )
                
                frames, masks = self.get_frames(support_video_id, shot_indices, category)
                all_support_frames.extend(frames)
                all_support_masks.extend(masks)
            
            # Get query masks for this category using the same frames
            if category in self.video_to_categories[query_video_id]:
                _, category_masks = self.get_frames(query_video_id, query_indices, category)
                all_query_masks.extend(category_masks)
            else:
                # If query video doesn't contain this category, use zero masks
                h, w = query_masks[0].shape[:2]
                zero_masks = [np.zeros((h, w), dtype=np.bool_) for _ in range(self.query_frames)]
                all_query_masks.extend(zero_masks)
            
            # Duplicate query frames for each category
            all_query_frames.extend(query_frames)

        # Apply transforms
        if self.transforms is not None:
            query_frames, query_masks = self.transforms(all_query_frames, all_query_masks)
            support_frames, support_masks = self.transforms(all_support_frames, all_support_masks)

        return query_frames, query_masks, support_frames, support_masks, [query_video_id]

    def __gettestitem__(self, idx):
        # 随机选择N个类别
        # if len(self.selected_categories) < self.num_ways:
        #     raise ValueError(f"Not enough categories for {self.num_ways}-way testing")
        # selected_categories = random.sample(self.selected_categories, self.num_ways)
        selected_categories = self.action_hierarchy.sample_fine_grained_episode(self.num_ways + 1)
        query_category = random.choice(selected_categories)
        selected_categories = selected_categories[:self.num_ways]
        # print(selected_categories)
        
        # 准备支持集和查询集数据
        all_support_frames = []
        all_support_masks = []
        all_query_frames = []
        all_query_masks = []
        
        # 首先随机选择一个查询类别
        # query_category = random.choice(selected_categories)
        
        # 获取该类别的所有视频
        category_videos = self.category_to_videos[query_category]
        
        # 确保有足够的视频用于support和query
        if len(category_videos) < self.num_shots + 1:
            # 如果视频不够，允许重复使用
            # support_video_ids = random.choices([v for v in category_videos if v != query_category], k=self.num_shots)
            query_video_id = random.choice(category_videos)
        else:
            # 随机选择support视频
            # support_video_ids = random.sample([v for v in category_videos if v != query_category], self.num_shots)
            # 从剩余视频中选择query视频
            # remaining_videos = [v for v in category_videos if v not in support_video_ids]
            query_video_id = random.choice(category_videos)
        
        # 获取query视频的所有帧
        valid_query_frames, all_query_frames_list = self.get_valid_frames(query_video_id, query_category)
        # 测试时使用所有帧
        query_indices = all_query_frames_list
        
        query_frames, query_masks = self.get_frames(query_video_id, query_indices, query_category)
        
        # 为每个类别获取支持集数据
        for category in selected_categories:
            # 获取该类别的所有视频
            category_videos = self.category_to_videos[category]
            
            # 获取支持集视频
            if category == query_category:
                support_video_ids = random.sample([v for v in category_videos if v != query_video_id], self.num_shots)
            else:
                if len(category_videos) < self.num_shots:
                    support_video_ids = random.choices(category_videos, k=self.num_shots)
                else:
                    support_video_ids = random.sample(category_videos, self.num_shots)
            
            # 获取支持集视频的帧
            for support_video_id in support_video_ids:
                valid_support_frames, all_support_frames_list = self.get_valid_frames(support_video_id, category)
                shot_indices = self.sample_frames_with_action_support(
                    valid_support_frames,
                    all_support_frames_list,
                    self.support_frames,
                )
                
                frames, masks = self.get_frames(support_video_id, shot_indices, category)
                all_support_frames.extend(frames)
                all_support_masks.extend(masks)
            
            # 获取查询帧的类别掩码
            if category in self.video_to_categories[query_video_id]:
                _, category_masks = self.get_frames(query_video_id, query_indices, category)
                all_query_masks.extend(category_masks)
            else:
                # 如果查询视频不包含此类别，使用零掩码
                h, w = query_masks[0].shape[:2]
                zero_masks = [np.zeros((h, w), dtype=np.bool_) for _ in range(len(query_indices))]
                all_query_masks.extend(zero_masks)
            
            # 为每个类别复制查询帧
            all_query_frames.extend(query_frames)
        
        # 应用数据增强
        if self.transforms is not None:
            query_frames, query_masks = self.transforms(all_query_frames, all_query_masks)
            support_frames, support_masks = self.transforms(all_support_frames, all_support_masks)
        
        return query_frames, query_masks, support_frames, support_masks, [query_video_id], selected_categories

    def __len__(self):
        # 返回一个足够大的数字，使得可以进行足够多的episode
        return 10000000  # 这个数字可以根据需要调整 


if __name__ == '__main__':
    
    from libs.dataset.transform import TrainTransform
    size = (241, 425)
    train_transform = TrainTransform(size)
    # dataset = MoVeDataset(train=False, valid=True, num_ways=2, num_shots=1, sample_method='hard', transforms=train_transform)
    # print(len(dataset))
    # query_frames, query_masks, support_frames, support_masks, query_video_id, selected_categories = dataset[0]
    # print(query_frames.shape, query_masks.shape, support_frames.shape, support_masks.shape, query_video_id, selected_categories)
    dataset = MoVeDataset(train=True, valid=False, num_ways=2, num_shots=1, sample_method='hard', transforms=train_transform, group=0)
    print(len(dataset))
    query_frames, query_masks, support_frames, support_masks, query_video_id = dataset[0]
    print(query_frames.shape, query_masks.shape, support_frames.shape, support_masks.shape)
