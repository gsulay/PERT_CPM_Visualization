import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class PERT_CPM:
    def __init__(self, df, label_replace = None, increment_by='days', date_offset=3, start_date=None):
        self.df = df
        self.label_replace = label_replace
        self.increment_by = increment_by
        self.date_offset = date_offset
        self.start_date = start_date

        self.activity_catalogue = self.init_matrix(self.df)
        self.activity_catalogue = self.get_forward_reference(self.activity_catalogue)
        self.activity_catalogue = self.forward_pass(self.activity_catalogue)
        self.activity_catalogue = self.backward_pass(self.activity_catalogue)
        self.activity_catalogue = self.get_slack(self.activity_catalogue)
        self.critical_path = self.calculate_critical_path(self.activity_catalogue)
        self.final_df = self.concat_df(self.activity_catalogue)

        if type(self.label_replace) != 'NoneType':
            self.final_df = self.replace_idx()
        self.create_gantt_chart()

    def init_matrix(self, df:pd.DataFrame):
        """Fills the back_ref and comp_time and inits the dataframe"""
        df = df.fillna('None')
        cols = df.columns.values
        cols = [col.strip().title() for col in cols]
        activity_catalogue = {}
        for col in cols:
            #Make dictionary per activity
            activity_catalogue[col] = {}
            selected_column = activity_catalogue[col]

            #Get Comp Time
            comp_time = df.loc['Completion Time', col]
            selected_column['comp_time'] = comp_time

            #init forward_ref
            selected_column['forward_ref'] = []

            #Get back_ref
            prereqs = df.loc['Prerequisites', col].split(',')   
            prereqs = [prereq.strip().title() for prereq in prereqs]    #Makes sure that the prereqs match with the col index
            selected_column['back_ref'] = prereqs

            #init_slack
            selected_column['slack'] = None

            #init matrix
            col_df = pd.DataFrame({ "start"     :[None]*2,
                                    "finish"    :[None]*2}, index=['early', 'late'])
            selected_column['df'] = col_df
        
        return activity_catalogue
        
    def get_forward_reference(self, activity_catalogue:dict):
        """iterates through each column and gets the prerequsite and add the columns index to the prerequisites' dict"""
        to_process = list(activity_catalogue.keys())
        for col in to_process:
            selected_column = activity_catalogue[col]
            prereqs = selected_column['back_ref']
            for prereq in prereqs:
                if prereq == 'None':
                    continue
                activity_catalogue[prereq]['forward_ref'].append(col)
        return activity_catalogue
    
    def forward_pass(self, activity_catalogue:dict):
        cols = list(activity_catalogue.keys())
        
        first_node = []
        #Get all columns who has no prereqs as first tree
        for col in cols:
            if activity_catalogue[col]['back_ref'][0] == 'None':
                first_node.append(col)
        #process first node
        for node in first_node:
            
            #Process first node
            selected_column = activity_catalogue[node]
            selected_column['df'].loc['early', 'start'] = 0
            selected_column['df'].loc['early', 'finish'] = selected_column['df'].loc['early', 'start']  + selected_column['comp_time']

        #Do forward pass !TODO: delete processed since its not really needed
        processed = []
        while True:
            #Check for already finished nodes
            to_remove = []
            for col in cols:
                selected_column = activity_catalogue[col]
                selected_df = selected_column['df']
                if selected_df.loc['early', 'finish'] != None:
                    processed.append(col)
                    to_remove.append(col)
            #Delete cols
            for col in to_remove:
                cols.remove(col)
            
            #Do forward pass in valid columns
            for col in cols:
                prereqs = activity_catalogue[col]['back_ref']
                is_ready = True
                for prereq in prereqs:  #Check if ready, if not skip
                    col_df = activity_catalogue[prereq]['df']
                    if col_df.loc['early', 'finish'] == None:
                        is_ready = False
                if is_ready != True:
                    continue

                es = max([activity_catalogue[p]['df'].loc['early', 'finish'] for p in prereqs])
                selected_df = activity_catalogue[col]['df']
                selected_df.loc['early', 'start']   = es
                selected_df.loc['early', 'finish']  = selected_df.loc['early', 'start']  + activity_catalogue[col]['comp_time']
            
            if len(cols) == 0:  #Breaks off the while loop when there is nothing left to process
                break
        return activity_catalogue

    def backward_pass(self, activity_catalogue:dict):
        cols = list(activity_catalogue.keys())
        
        end_nodes = []
        #Get all columns who has no forward ref as leaf
        for col in cols:
            if len(activity_catalogue[col]['forward_ref']) == 0:
                end_nodes.append(col)

        #process end nodes
        lf = max([activity_catalogue[node]['df'].loc['early', 'finish'] for node in end_nodes]) #Get max 
        for node in end_nodes:
            selected_df = activity_catalogue[node]['df']
            selected_df.loc['late', 'finish']   = lf
            selected_df.loc['late', 'start']    = selected_df.loc['late', 'finish'] - activity_catalogue[node]['comp_time']
        
        #Backward pass
        while True:
            #Check for already finished nodes and delete from cols
            to_remove = []
            for col in cols:
                selected_df = activity_catalogue[col]['df']
                if selected_df.loc['late', 'start'] != None:
                    to_remove.append(col)
            for col in to_remove:   #Delete from cols
                cols.remove(col)
            
            #Backward pass on valid columns
            for col in cols:
                forward_refs = activity_catalogue[col]['forward_ref']
                is_ready = True
                for ref in forward_refs:  #Check if ready, if not skip
                    col_df = activity_catalogue[ref]['df']
                    if col_df.loc['late', 'start'] == None:
                        is_ready = False
                if is_ready != True:
                    continue

                lf = min([activity_catalogue[f]['df'].loc['late', 'start'] for f in forward_refs])
                selected_df = activity_catalogue[col]['df']
                selected_df.loc['late', 'finish']   = lf
                selected_df.loc['late', 'start']    = selected_df.loc['late', 'finish'] - activity_catalogue[col]['comp_time']
            
            if len(cols) == 0:  #Breaks off the while loop when there is nothing left to process
                break
        
        return activity_catalogue
    
    def get_slack(self, activity_catalogue:dict):
        cols = list(activity_catalogue.keys())
        for col in cols:
            selected_df = activity_catalogue[col]['df']
            activity_catalogue[col]['slack'] = selected_df.loc["late", "start"] - selected_df.loc["early", "start"]
        return activity_catalogue
    
    def calculate_critical_path(self, activity_catalogue):
        # Create a directed acyclic graph
        G = nx.DiGraph()

        # Add nodes to the graph
        for activity, details in activity_catalogue.items():
            G.add_node(activity)

        # Add edges to the graph
        for activity, details in activity_catalogue.items():
            forward_refs = details['forward_ref']
            for ref in forward_refs:
                G.add_edge(activity, ref)

        # Find the critical activities based on longest duration or zero slack
        critical_path = []
        for activity, details in activity_catalogue.items():
            slack = details['slack']
            if slack == 0.0:
                critical_path.append(activity)

        return critical_path

    def concat_df(self, activity_catalogue):
        cols = list(activity_catalogue.keys())
        df_list = []
        for col in cols:
            sample_df = activity_catalogue[col]['df']
            es = sample_df.loc['early', 'start']
            ef = sample_df.loc['early', 'finish']
            ls = sample_df.loc['late', 'start']
            lf = sample_df.loc['late', 'finish']

            new_df = pd.DataFrame({ 'Early Start':[es],
                                    'Early Finish':[ef],
                                    'Late Start':[ls],
                                    'Late Finish':[lf]}, index=[col])
            df_list.append(new_df)

        final_df = pd.concat(df_list)
        return final_df

    def create_gantt_chart(self):
        start_date = self.start_date
        increment_by = self.increment_by
        date_offset = self.date_offset
        """if there is start date, format = <YYYY-MM-DD>"""

        df = self.final_df.copy()
        fig, ax = plt.subplots(figsize=(25, 15))

        # Plot horizontal bars for each activity
        for i, activity in enumerate(df.index):
            start = df.loc[activity, 'Early Start']
            duration = df.loc[activity, 'Late Finish'] - start
            es = start
            ls = df.loc[activity, 'Late Start']
            ef = df.loc[activity, 'Early Finish']
            lf = df.loc[activity, 'Late Finish']
            ax.barh(y=i, left=start, width=duration, height=0.5, color='lightblue', edgecolor='black')
            ax.text(start + duration, i, f"ES:{int(es)} LS:{int(ls)}\nEF:{int(ef)} LF:{int(lf)}", ha='left', va='center')

        # Set y-axis ticks and labels
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)

        # Set x-axis label and limits
        ax.set_xlabel('Time')
        ax.set_xlim(0, df['Late Finish'].max() + 1)

        #Set x-ticks if there is a start_date
        if start_date is not None:
            xticks = np.arange(0, df['Late Finish'].max() + 1, date_offset)   #sets date_offset
            ax.set_xticks(xticks)

            start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
            if increment_by == 'days':
                xticks_date = [start_date + pd.DateOffset(days=i) for i in xticks]
            elif increment_by == 'months':
                xticks_date = [start_date + pd.DateOffset(months=i) for i in xticks]
            elif increment_by == 'weeks':
                xticks_date = [start_date + pd.DateOffset(weeks=i) for i in xticks]
            else:
                raise ValueError("No or wrong increment_by provided. Only accepts ['days','months', 'weeks']")
            xticks_date = [i.strftime('%Y-%m-%d') for i in xticks_date]
            ax.set_xticklabels(xticks_date, rotation = 90, ha = 'center')

        # Set chart title
        ax.set_title('PERT/CPM Gantt Chart')

        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Invert y-axis to show activities from top to bottom
        ax.invert_yaxis()

        # Show gridlines
        ax.grid(True, axis='x', linestyle='--', color='gray', alpha=0.7)

        plt.tight_layout()

        self.fig = fig
    
    def replace_idx(self):
        df = self.final_df
        replace_format = {k:v for k, v in zip(self.label_replace.iloc[:,0].to_list(), self.label_replace.iloc[:,1].to_list())}
        final_df = df.rename(index=replace_format)
        self.replace_format = replace_format
        return final_df
    
    def export(self):
        increment_by  = self.increment_by
        start_date = self.start_date
        start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        def increment_from_start(val):
            try:
                if increment_by == 'days':
                    val = start_date + pd.DateOffset(days=val)
                elif increment_by == 'months':
                    val = start_date + pd.DateOffset(months=val) 
                elif increment_by == 'weeks':
                    val = start_date + pd.DateOffset(weeks=val) 
                else:
                    raise ValueError("No or wrong increment_by provided. Only accepts ['days','months', 'weeks']")
                return val.strftime('%Y-%m-%d')
            except TypeError:
                print("Error on Val: ", val)
            
        with pd.ExcelWriter('PERT_CPM.xlsx') as f:
            df1 = self.final_df
            #Get Slack
            slack = df1.loc[:, "Late Start"] - df1.loc[:, 'Early Start']
            #Gets the completion time and prereqs
            replace_idx = {k:v for v, k in self.replace_format.items()}
            comp_time = []
            prereqs_final = []
            for idx in df1.index.values:
                col = replace_idx[idx]
                comp_time.append(self.activity_catalogue[col]['comp_time'])

                prereqs = self.activity_catalogue[col]['back_ref']
                prereqs_name = []
                for p in prereqs:
                    if (p == None or p=='None'):
                        continue
                    name = self.replace_format[p]
                    prereqs_name.append(name)
                prereqs_final.append(prereqs_name)
            
            prereqs_final = [",".join(i) for i in prereqs_final]

            #Sheet with dates
            if self.start_date is not None:
                #Increments values from start date
                df = self.final_df.copy()
                df2 = df.applymap(increment_from_start)
                df2['Completion Time'] = comp_time
                df2['Slack'] = slack
                df2['Prerequisites'] = prereqs_final
                df2.to_excel(f, sheet_name="PERT CPM Dates")

            df1['Completion Time'] = comp_time
            df1['Slack'] = slack
            df1['Prerequisites'] = prereqs_final
            
            
            df1.to_excel(f, sheet_name="PERT CPM Table")
    
            df3 = self.df
            df3.to_excel(f, sheet_name="Prerequisites")

            df4 = self.label_replace
            df4 = df4.set_index('Index')
            df4.to_excel(f, sheet_name="Labels")
        self.fig.savefig("PERT CPM Chart.png")
