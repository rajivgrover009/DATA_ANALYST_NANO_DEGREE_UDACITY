#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
import matplotlib.path as mt
import geocoder
from geopy.point import Point

from matplotlib.path import Path
from geopy.geocoders import Nominatim
from collections import defaultdict




expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road",
            "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping = { "St": "Street",
            "St.": "Street",
            "Ave": "Avenue",
            "Rd." : "Road",
            "rd" : "Road",
            "Rd" : "Road",
            "E" : "East",
            "S" : "South",
            "S.": "South",
            "E.": "East"}
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
street_types = defaultdict(set)
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]
VALIDZIPS=['80010','80011','80013','80014','80015','80018','80019','80012','80016','80017','80042',
            '80044','80045','80046','80047','80040','80041']

#method to check if given element is a street
#Returns true if it is a street else returns False
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")
global count

#method to check if post code is valid
#Returns true if it is a valid post code else returns False
def is_valid_postcode(ele):
    if ele in VALIDZIPS:
        return True
    else:
        return False


#method to convert name in to appropriate format to bring uniformity
def update_name(name, mapping):

    for v in mapping:
        if re.search(v,name):
            if v in ["E","S","S.","E."]: #check if current key is one of the the given keys
                if len(name.split(' ')[0])<=2:  # if the first element of the name after splititng is E, E. etc
                    name=mapping[v]+" "+name[len(v):] # prefix the correct name
        else:
            if re.search(v,name):
                name=name[:name.find(v)]+mapping[v] # suffix the correct name

    return name

#audit the street type
def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

# Function to clean the k values that have : in them. It also does some additional check for city,county,state and
# postcode

def clean_colon_k_values(tag):
    valid=True
    value = tag.get('v')
    tild_list=[]
    if is_street_name(tag):
        audit_street_type(street_types, tag.attrib['v'])
        value=update_name(value, mapping)
    ktext=tag.get('k')
    ktext_after_colon= ktext[ktext.find(':')+1:]

    if ktext_after_colon.find(':')==-1:
        if ktext_after_colon.strip(' ')=="postcode":
            if is_valid_postcode(value)==False: # don't include invalid addresses
                valid=False
        if ktext_after_colon.strip(' ')=="city": # exclude cities which are not aurora
            if value!="Aurora":
                valid=False
        if ktext_after_colon.strip(' ')=="state": # fix state to a standard name
            if value=="CO" or value == "Co":
                value= "Colorado"
        if ktext_after_colon.strip(' ')=="county_name": # fix county_name key value
            ktext_after_colon="county"
            if value.find(',')>0: # standardized county name , Remove the state after comma
                value=value[:value.find(',')]
        if ktext_after_colon.strip(' ') =="tlid":
            value=value.split(':')  # put the items in a list


    return ktext_after_colon,value,valid

def shape_element(element):
    node = {}
    pos=[]
    created={}
    addr={}
    tiger={}
    gnis ={}
    refs=[]
    valid_node=True
    if element.tag == "node" or element.tag == "way" :
        if element.tag == "node":
            node['type']='node'
        else:
            node['type']='way'
        if element.get('id')!=None: #check if attribute exists
            node['id']=element.get('id')
        if element.get('type')!=None: #check if attribute exists
            node['type']=element.get('type')
        if element.get('visible')!=None: #check if attribute exists
            node['visible']=element.get('visible')
        if element.get('lat')!=None: #check if attribute exists
            pos.append(float(element.get('lat')))
            if element.get('lon')!=None: #check if attribute exists
                pos.append(float(element.get('lon')))
            node['pos']=pos #put the list in node with key as pos
        for ele in CREATED:
            created[ele]=element.get(ele)
        node['created']=created

        for tag in list(element.iter('tag')): #create key value pair for all tags
            if re.search(problemchars,tag.get('k')):    # ignore tags with problematic chars
                continue
            # clean the field with value addr: and gnis: and tiger:
            if tag.get('k').startswith("addr:") or tag.get('k').startswith("gnis:") or tag.get('k').startswith("tiger:"):
                ktext,value,valid_node=clean_colon_k_values(tag)
                if tag.get('k').startswith("tiger:"):
                    tiger[ktext]=value  # add in to tiger dictionary first
                elif tag.get('k').startswith("gnis:"):
                    gnis[ktext]=value  # add in to gnis dictionary first
                else:
                    addr[ktext]=value # create directly a key value pair
                if valid_node==False: # check if clean_colon_k_values function return a valid node
                    break
            else:
                node[tag.get('k')]=tag.get('v')
        for tag in list(element.iter('nd')): # loop through nodes with nd
            refs.append(tag.get('ref'))
        if len(addr)>0: # # check if there are any items in addr dictionary
            node['address']=addr
        if len(refs)>0: # check if there are any items in refs list
            node['node_refs']=refs
        if len(gnis)>0:    # check if there are any items in gnis dictionary
            node['gnis']=gnis
        if len(tiger)>0:    # check if there are any items in tiger dictionary
            node['tiger']=tiger

    return node,valid_node


def process_map(file_in, pretty = False):
    # You do not need to change this file
    file_out = "{0}.json".format(file_in)
    file_out_invalid = "{0}_invalid.json".format(file_in) #output invalid nodes in a separate file
    data = []
    invalid_data = []
    count = 0
    with codecs.open(file_out, "w") as fo:
        with codecs.open(file_out_invalid, "w") as fo_invalid: # open file to write invalid nodes
            for _, element in ET.iterparse(file_in):
                el,is_valid = shape_element(element)
                if el and is_valid:
                    data.append(el)
                    if pretty:
                        fo.write(json.dumps(el, indent=2)+"\n")
                    else:
                        fo.write(json.dumps(el) + "\n")
                elif el and is_valid==False: # if node isn ot valid ,write in to separate file
                    invalid_data.append(el)
                    if pretty:
                        fo_invalid.write(json.dumps(el, indent=2)+"\n")
                    else:
                        fo_invalid.write(json.dumps(el) + "\n")
                else:
                    pass


    return data

#Call function
process_map("E:\DataScienceWithR\Nano Degree Udacity\Projects\Project 3\map",False)
