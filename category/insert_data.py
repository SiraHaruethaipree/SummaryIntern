from huggingface_hub import HfApi, list_models
import mysql.connector
import pandas as pd
import json
import numpy as np

def get_subtask_to_index_dict(cursor):
    print("Getting subtask dictionary to index")

    sql_query = """
    select * from modelsubtasktype
    """

    cursor.execute(sql_query)
    field_names = [i[0] for i in cursor.description]
    sub = cursor.fetchall()
    sub_df = pd.DataFrame(sub, columns = field_names )
    subtask_to_code_dict = sub_df.set_index('name')['code'].to_dict()
    print("Successfully get dictionary to index")
    print(subtask_to_code_dict)
    return subtask_to_code_dict


def get_huggingface_model():
    print("Getting models from Huggingface: downloading...")
    models = list_models(full=True,
                         cardData=True) #Must specify this to get the whole full list of information about last modified, author, siblings

    model_lists = list(models)
    print("Getting models from Huggingface: successfully getting models...")
    return model_lists


def convert_to_df(model_lists):
    model_df = pd.DataFrame(d.__dict__ for d in model_lists)
    return model_df


def convert_subtask(name, subtask_to_code_dict):
    if name not in subtask_to_code_dict.keys():
        value = 99

    else:
        value = subtask_to_code_dict.get(name)

    return value


def create_injection_df(model_df, subtask_to_code_dict):

    metadata_cols = ['_id', 'tags', 'cardData', 'private' ]


    final_df = pd.DataFrame({
    'modelname' : model_df['modelId'],
    'modelsourcetypecode':  [1 for item in range(len(model_df))],
    'modelsubtasktypecode':  np.vectorize(convert_subtask)(model_df['pipeline_tag'],subtask_to_code_dict),                  
    'modelmodifieddate': model_df['lastModified'],
    'author': model_df['author'],
    'library_name': model_df['library_name'],
    'numberlikes': model_df['likes'],
    'numberdownloads': model_df['downloads'],
    'metadata': [model_df[metadata_cols].iloc[item, :].to_dict() for item in range(len(model_df))],
    
})
    final_df['metadata'] = final_df['metadata'].apply(json.dumps)
    return final_df


def check_to_insert_or_upate(cursor, final_df):
    all_insert_query = """
    insert into modelheader (modelname, modelsourcetypecode, modelsubtasktypecode,
       modelmodifieddate, author, library_name, numberlikes,
       numberdownloads, metadata, createddate, createdby)


    value (%s, %s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(), 'Poon')
    """


    all_update_query = """
        update modelheader
        set modelmodifieddate = %s, numberlikes = %s, numberdownloads = %s, updateddate = current_timestamp(), updatedby = %s, 
        where modelname = %s
        """
    
    for data in [tuple(x) for x in final_df.to_records(index=False)]:
        cursor.execute("SELECT * FROM modelheader WHERE modelname = %s", (data[0],))
        existing_row = cursor.fetchone()
        # print(cursor.fetchone())
        if existing_row:
            # Row exists, update it3
            # print('alreadu have')
            cursor.execute(all_update_query, (data[3], int(data[6]), int(data[7]), 'Poon', data[0]))
        else:
            # Row does not exist, insert it
            # print("not having")
            # print(data)
            cursor.execute(all_insert_query, (data[0], int(data[1]), int(data[2]), data[3], data[4], data[5], int(data[6]), int(data[7]), json.dumps(data[8])))
            
    myconn.commit()


if __name__ == "__main__":
    #Create the connection object   
    myconn = mysql.connector.connect(host = "172.17.101.178", 
                                    user = "root",
                                    passwd = "2CWTL9jREpp%WbqS", 
                                    port = 3326,
                                    database = "hf_model2")  
        
    cursor = myconn.cursor()

    subtask_to_code_dict =  get_subtask_to_index_dict()
    model_lists = get_huggingface_model()
    model_df = convert_to_df(model_lists=model_lists)
    final_df = create_injection_df(model_df=model_df,
                                   subtask_to_code_dict=subtask_to_code_dict)


    final_df.to_csv("model_df_to_insert_databaset")

    check_to_insert_or_upate(cursor=cursor,
                             final_df=final_df)
