import requests
from tqdm import tqdm
from collections import defaultdict
import json


def clean_up_description(description):
    description = description + " "

    ##### extra adj Pure #####
    if description.startswith("Pure "):
        description = description.replace("Pure ", "")
    ##### fix typo #####
    if description.startswith("Mercurycombines"):
        description = description.replace("Mercurycombines", "Mercury combines")
    
    name_special_case_list = [
        '17-Hydroxy-6-methylpregna-3,6-diene-3,20-dione. ',
        '5-Thymidylic acid. ',
        "5'-S-(3-Amino-3-carboxypropyl)-5'-thioadenosine. ",
        "Guanosine 5'-(trihydrogen diphosphate), monoanhydride with phosphorothioic acid. ",
        "5'-Uridylic acid. ",
        "5'-Adenylic acid, ",
        "Uridine 5'-(tetrahydrogen triphosphate). ",
        "Inosine 5'-Monophosphate. ",
        "Pivaloyloxymethyl butyrate (AN-9), ",
        "4-Amino-5-cyano-7-(D-ribofuranosyl)-7H- pyrrolo(2,3-d)pyrimidine. ",
        "Cardamonin (also known as Dihydroxymethoxychalcone), ",
    ]

    ##### a special case #####
    description = description.replace("17-Hydroxy-6-methylpregna-3,6-diene-3,20-dione. ", "17-Hydroxy-6-methylpregna-3,6-diene-3,20-dione is ")

    ##### a special case #####
    description = description.replace("5-Thymidylic acid. ", "5-Thymidylic acid. is ")

    ##### a special case #####
    description = description.replace("5'-S-(3-Amino-3-carboxypropyl)-5'-thioadenosine. ", "5'-S-(3-Amino-3-carboxypropyl)-5'-thioadenosine. is ")

    ##### a special case #####
    description = description.replace("Guanosine 5'-(trihydrogen diphosphate), monoanhydride with phosphorothioic acid. ", "Guanosine 5'-(trihydrogen diphosphate), monoanhydride with phosphorothioic acid is ")

    ##### a special case #####
    description = description.replace("5'-Uridylic acid. ", "5'-Uridylic acid is ")

    ##### a special case #####
    description = description.replace("5'-Adenylic acid, ", "5'-Adenylic acid is ")

    ##### a special case #####
    description = description.replace("Uridine 5'-(tetrahydrogen triphosphate). ", "Uridine 5'-(tetrahydrogen triphosphate). is ")

    ##### a special case #####
    description = description.replace("Inosine 5'-Monophosphate. ", "Inosine 5'-Monophosphate. is ")

    ##### a special case #####
    description = description.replace("Pivaloyloxymethyl butyrate (AN-9), ", "Pivaloyloxymethyl butyrate (AN-9) is ")

    ##### a special case #####
    description = description.replace("4-Amino-5-cyano-7-(D-ribofuranosyl)-7H- pyrrolo(2,3-d)pyrimidine. ", "4-Amino-5-cyano-7-(D-ribofuranosyl)-7H- pyrrolo(2,3-d)pyrimidine is ")

    ##### a special case #####
    description = description.replace("Cardamonin (also known as Dihydroxymethoxychalcone), ", "Cardamonin (also known as Dihydroxymethoxychalcone) is ")

    ##### a special case #####
    description = description.replace("Lithium has been used to treat ", "Lithium is ")

    ##### a special case #####
    description = description.replace("4,4'-Methylenebis ", "4,4'-Methylenebis is ")

    ##### a special case #####
    description = description.replace("2,3,7,8-Tetrachlorodibenzo-p-dioxin", "2,3,7,8-Tetrachlorodibenzo-p-dioxin is ")

    ##### a special case #####
    description = description.replace("Exposure to 2,4,5-trichlorophenol ", "2,4,5-Trichlorophenol exposure ")

    index = 0
    L = len(description)
    if description.startswith('C.I. '):
        start_index = len('C.I. ')
    elif description.startswith('Nectriapyrone. D '):
        start_index = len('Nectriapyrone. D ')
    elif description.startswith('Salmonella enterica sv. Minnesota LPS core oligosaccharide'):
        start_index = len('Salmonella enterica sv. Minnesota LPS core oligosaccharide')
    else:
        start_index = 0
    for index in range(start_index, L - 1):
        if index < L-2:
            if description[index] == '.' and description[index+1] == ' ' and 'A' <= description[index+2] <= 'Z':
                break
        elif index == L - 2:
            break
    
    first_sentence = description[:index+1]
    return first_sentence


def extract_name(name_raw, description):
    first_sentence = clean_up_description(description)

    splitter = '  --  --  '
    if ' are ' in first_sentence or ' were ' in first_sentence:
        replaced_words = 'These molecules'
    else:
        replaced_words = 'This molecule'

    first_sentence = first_sentence.replace(' is ', splitter)
    first_sentence = first_sentence.replace(' are ', splitter)
    first_sentence = first_sentence.replace(' was ', splitter)
    first_sentence = first_sentence.replace(' were ', splitter)
    first_sentence = first_sentence.replace(' appears ', splitter)
    first_sentence = first_sentence.replace(' occurs ', splitter)
    first_sentence = first_sentence.replace(' stands for ', splitter)
    first_sentence = first_sentence.replace(' belongs to ', splitter)
    first_sentence = first_sentence.replace(' exists ', splitter) # only for CID=11443
    first_sentence = first_sentence.replace(' has been used in trials ', splitter)
    first_sentence = first_sentence.replace(' has been investigated ', splitter)
    first_sentence = first_sentence.replace(' has many uses ', splitter)
    
    if splitter in first_sentence:
        extracted_name = first_sentence.split(splitter, 1)[0]
    elif first_sentence.startswith(name_raw):
        extracted_name = name_raw
    elif name_raw in first_sentence:
        extracted_name = name_raw
        extracted_name = None
        print("=====", name_raw)
        print("first sentence: ", first_sentence)
        # print()
    else:
        extracted_name = None

    if extracted_name is not None:
        extracted_description = description.replace(extracted_name, replaced_words)
    else:
        extracted_description = description

    return extracted_name, extracted_description, first_sentence


if __name__ == "__main__":
    total_page_num = 290
    # Please put your own dataset path here
    datasets_home_folder = "../../../Datasets"

    PubChemSTM_datasets_description_home_folder = "{}/step_01_PubChemSTM_description".format(datasets_home_folder)
    valid_CID_list = set()
    CID2name_raw, CID2name_extracted = defaultdict(list), defaultdict(list)
    CID2text_raw, CID2text_extracted = defaultdict(list), defaultdict(list)

    for page_index in tqdm(range(total_page_num)):
        page_num = page_index + 1
        compound_description_file_name = "Compound_description_{}.txt".format(page_num)
        f_out = open("{}/{}".format(PubChemSTM_datasets_description_home_folder, compound_description_file_name), "w")
        
        description_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/json?heading_type=Compound&heading=Record+Description&page={}".format(page_num)
        description_data = requests.get(description_url).json()

        description_data = description_data["Annotations"]
        assert description_data["Page"] == page_num
        assert description_data["TotalPages"] == total_page_num
        
        record_list = description_data["Annotation"]
        
        for record in record_list:
            try:
                CID = record["LinkedRecords"]["CID"][0]
                if "Name" in record:
                    name_raw = record["Name"]
                    CID2name_raw[CID].append(name_raw)
                else:
                    name_raw = None

                data_list = record["Data"]
                for data in data_list:
                    description = data["Value"]["StringWithMarkup"][0]["String"].strip()
                    
                    extracted_name, extracted_description, first_sentence = extract_name(name_raw, description)
                    if extracted_name is not None:
                        CID2name_extracted[CID].append(extracted_name)

                    CID_special_case_list = [45266824, 11683, 3759, 9700, 439155, 135398675, 135563708, 6030, 10238, 6133, 135398640, 77918, 60748, 11824, 641785, 11125, 7543, 15625, 7271]

                    ##### only for debugging #####
                    if CID in CID_special_case_list:
                        print("page: {}\tCID: {}".format(page_index, CID))                
                        if "Name" in record:
                            print('yes-name')
                            name = record["Name"]
                            print('name:', name)
                        else:
                            print('no-name')
                        print('extracted name:', extracted_name)
                        print("first_sentence:", first_sentence)
                        print("extracted_description:", extracted_description)
                        print("description:", description)
                        print() 
                        
                    CID2text_raw[CID].append(description)
                    CID2text_extracted[CID].append(extracted_description)

                    valid_CID_list.add(CID)
                    f_out.write("{}\n".format(CID))
                    f_out.write("{}\n\n".format(extracted_description))
            except:
                # print("===\n", record)
                # print("missing page: {}\tSourceName: {}\tSourceID: {}".format(page_index, record['SourceName'], record['SourceID']))
                continue
            
    valid_CID_list = list(set(valid_CID_list))
    valid_CID_list = sorted(valid_CID_list)
    # print("valid CID list: {}".format(valid_CID_list))
    print("Total CID (with raw name) {}".format(len(CID2name_raw)))
    print("Total CID (with extracted name) {}".format(len(CID2name_extracted)))
    print("Total CID {}".format(len(valid_CID_list)))
    
    with open("{}/PubChemSTM_data/raw/CID2name_raw.json".format(datasets_home_folder), "w") as f:
        json.dump(CID2name_raw, f)
    
    with open("{}/PubChemSTM_data/raw/CID2name.json".format(datasets_home_folder), "w") as f:
        json.dump(CID2name_extracted, f)

    with open("{}/PubChemSTM_data/raw/CID2text_raw.json".format(datasets_home_folder), "w") as f:
        json.dump(CID2text_raw, f)

    with open("{}/PubChemSTM_data/raw/CID2text.json".format(datasets_home_folder), "w") as f:
        json.dump(CID2text_extracted, f)