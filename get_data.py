import pandas as pd
import numpy as np

def get_data(step = 10,force_reload=False,num_users=92):
    res = []
    for i in range(num_users):
        file_name = f"./data/user{i+1}_pickled"
        df = None
        try :
            if force_reload:
                print("force reload?")
                raise Exception("Reload")
            df = pd.read_pickle(file_name)
        except:
            print("loading?")
            df = pd.read_csv(f"./data/user{i+1}.csv")

            current_week = 1
            last_day = None
            last_session_time = None
            def discretize(r):
                nonlocal last_day,last_session_time,current_week
                row = r.copy()
                day = row["day_of_week"]
                session_time = row["Session_Entry"]
                if last_day is not None and last_day == day and session_time <=session_time:
                    current_week += 1
                elif last_day is not None and day < last_day:
                    current_week += 1
                row["Session_Entry"] = int(((current_week-1) * 7 +day) * 1440 + (session_time-1))
                last_day = day
                last_session_time = session_time
                return row

            df[["Session_Entry","day_of_week"]] = df[["Session_Entry","day_of_week"]].apply(discretize,axis=1)
            df.drop('day_of_week', axis=1, inplace=True)
            original_df = df.copy()

            df.index = df["Session_Entry"]
            last = int(df.iloc[-1]["Session_Time"]-1)
            df = df.reindex(np.arange(df.Session_Entry.min(), df.Session_Entry.max()+last + 1,step), fill_value=2957)

            for i,row in original_df.iterrows():
                # df.loc[row["Session_Entry"]:row["Session_Entry"]+row["Session_Time"]] = row
                start = row["Session_Entry"]
                end= row["Session_Entry"]+row["Session_Time"]-2
                temp = df.loc[start:end].copy()
                temp = temp.apply(lambda x: row,axis=1)
                df.loc[start:end] = temp
            df.drop('Session_Time', axis=1, inplace=True)
            df["Session_Entry"] = df.index
            min_ = df.index.min()
            df.index = df.index.map(lambda x: int(x-min_))
            df.to_pickle(file_name)
        res.append(df)
    return res