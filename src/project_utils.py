from __future__ import absolute_import
import os
import sys
import errno, copy
import os.path as osp
import more_itertools as mit
import torch, torch.nn as nn
import numpy as np
import scipy.io as sio

from torch.utils.data.sampler import Sampler


def read_Duke_attributes(attribute_file_path):
    attributes = [
        "age",
        "backpack",
        "bag",
        "boots",
        "clothes",
        "down",
        "gender",
        "hair",
        "handbag",
        "hat",
        "shoes",
        # "top", # patched up by changing to up
        "up",
        {"upcloth": [
            "upblack",
            "upblue",
            "upbrown",
            "upgray",
            "upgreen",
            "uppurple",
            "upred",
            "upwhite",
            "upyellow",
        ]},
        {"downcloth": [
            "downblack",
            "downblue",
            "downbrown",
            "downgray",
            "downgreen",
            "downpink",
            "downpurple",
            "downred",
            "downwhite",
            "downyellow",
        ]},
    ]

    attribute2index = {}
    for index, item in enumerate(attributes):
        key = item
        if isinstance(item, dict):
            key = list(item.keys())[0]
        attribute2index[key] = index

    num_options_per_attributes = {
        "age": ["young", "teen", "adult", "old"],
        "backpack": ["no", "yes"],
        "bag": ["no", "yes"],
        "boots": ["no", "yes"],
        "clothes": ["dress", "pants"],
        "down": ["long", "short"],
        "downcloth": [
            "downblack",
            "downblue",
            "downbrown",
            "downgray",
            "downgreen",
            "downpink",
            "downpurple",
            "downred",
            "downwhite",
            "downyellow",
        ],
        "gender": ["male", "female"],
        "hair": ["short", "long"],
        "handbag": ["no", "yes"],
        "hat": ["no", "yes"],
        "shoes": ["dark", "light"],
        # "top": ["short", "long"],
        "up": ["long", "short"],
        "upcloth": [
            "upblack",
            "upblue",
            "upbrown",
            "upgray",
            "upgreen",
            "uppurple",
            "upred",
            "upwhite",
            "upyellow",
        ],
    }

    def parse_attributes_by_id(attributes):
        # get the order of attributes in the buffer (from dtype)
        attribute_names = attributes.dtype.names
        attribute2index = {
            attribute: index
            for index, attribute in enumerate(attribute_names)
            if attribute != "image_index"
        }

        # get person id and arrange by index
        current_attributes = attributes.item()
        all_ids = current_attributes[-1]
        id2index = {id_: index for index, id_ in enumerate(all_ids)}

        # collect attributes for each ID
        attributes_byID = {}
        for attribute_name, attribute_index in attribute2index.items():
            # each attribute values are stored as a row in the attribute annotation file
            current_attribute_values = current_attributes[attribute_index]

            for id_, id_index in id2index.items():
                if id_ not in attributes_byID:
                    attributes_byID[id_] = {}

                attribute_value = current_attribute_values[id_index]

                # patch for up vs. top
                if attribute_name == "top":
                    attribute_name = "up"
                    attribute_value = (attribute_value % 2) + 1

                attributes_byID[id_][attribute_name] = (
                    attribute_value - 1
                )  # 0-based values

        return attributes_byID

    def merge_ID_attributes(train_attributes, test_attributes):
        all_ID_attributes = copy.deepcopy(train_attributes)

        for ID, val in test_attributes.items():
            assert (
                ID not in train_attributes.keys()
            ), "attribute merge: test ID {} already exists in train".format(ID)

            all_ID_attributes[ID] = val

        return all_ID_attributes


    root_dir, filename = osp.split(attribute_file_path)
    root_key = osp.splitext(filename)[0]
    buffer_file_path = osp.join(root_dir, root_key + "_attribute_cache.pth")

    if not osp.exists(buffer_file_path):
        print(buffer_file_path + " does not exist!, reading the attributes")
        att_file_contents = sio.loadmat(
            attribute_file_path, squeeze_me=True
        )  # squeeze 1-dim elements

        attributes = att_file_contents[root_key]

        train_attributes = attributes.item()[0]
        test_attributes = attributes.item()[1]

        train_attributes_byID = parse_attributes_by_id(train_attributes)
        test_attributes_byID = parse_attributes_by_id(test_attributes)
        print('train', len(train_attributes_byID), 'test', len(test_attributes_byID))
        all_ID_attributes = merge_ID_attributes(
            train_attributes_byID, test_attributes_byID
        )

        # writing attribute cache
        print("saving the attributes to cache " + buffer_file_path)
        torch.save(all_ID_attributes, buffer_file_path)

    else:
        print("reading the attributes from cache " + buffer_file_path)
        all_ID_attributes = torch.load(buffer_file_path)
    
    return all_ID_attributes


def read_Mars_attributes(attributes_file):
    pass


def shortlist_Mars_on_attribute(data_source, attribute, value):
    attributes_file = ''
    all_attributes = read_Mars_attributes(attributes_file)

    for img_path, pid, camid in data_source:
        pass
         

def shortlist_Duke_on_attribute(data_source, attribute, value):
    attributes_file = '/media/data1/datasets/personreid/dukemtmc-reid/attributes/duke_attribute.mat'
    all_attributes = read_Duke_attributes(attributes_file)

    # collect all indices where the attribute has particular value
    relevant_indices = []
    not_found_pids = []
    for index, (img_path, pid, camid) in enumerate(data_source):
        stringify_pid = "%04d" % pid

        if stringify_pid not in all_attributes:
            not_found_pids.append(stringify_pid)
            continue

        current_pid_attributes = all_attributes[stringify_pid]
        if current_pid_attributes[attribute] == value:
            relevant_indices.append(index)
    
    print('not found pids : {}'.format(not_found_pids))
    print(len(relevant_indices), ' sampled out of ', len(data_source))   
    return relevant_indices, all_attributes      


class AttributeBasedSampler(Sampler):
    def __init__(self, data_source, attribute, value, dataset_name):
        super().__init__(data_source)

        if dataset_name == 'mars':
            relevant_indices, _ = shortlist_Mars_on_attribute(data_source, attribute, value)
        elif dataset_name == 'dukemtmcvidreid':
            relevant_indices, _ = shortlist_Duke_on_attribute(data_source, attribute, value)
        else: 
            assert False, 'unknown dataset ' + dataset_name

        # get the indices of data instances which has
        # attribute's value as value
        self.instance_indices = relevant_indices

    def __len__(self):
        return len(self.instance_indices)

    def __iter__(self):
        return iter(self.instance_indices)
