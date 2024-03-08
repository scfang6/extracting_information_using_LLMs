from config1 import df
import pandas as pd
import numpy as np
import re

def generate_llm_df(gold_headers, match_headers, updated_letters):
    
    df2 = df[gold_headers].copy()
    df2_array = df2.values
    llm_array = np.zeros(df2_array.shape)


    for i,result in enumerate(updated_letters):

        result = result.replace('\"','')
        result = result.lower()

        for j, header in enumerate(match_headers):
            keyword = header.lower()+':'
            regex = rf'{re.escape(keyword)}\s*(\w+)'
            matches = re.findall(regex, result)

            if len(matches)>0:
                if matches[0] == 'true':
                    llm_array[i,j]=1.0

    df3 = pd.DataFrame(llm_array,columns=gold_headers)
    return df3