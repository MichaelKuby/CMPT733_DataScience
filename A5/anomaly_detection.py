import pandas as pd
import ast
from sklearn.cluster import KMeans

class AnomalyDetection():
    def cat2Num(self, df, indices):
        """ 
            Input: $df represents a DataFrame with two columns: "id" and "features"
                    $indices represents which dimensions in $features are categorical features, 
                    e.g., indices = [0, 1] denotes that the first two dimensions are categorical features.
                    
            Output: Return a new DataFrame that updates the "features" column with one-hot encoding. 
            
        """
        num_cat = len(indices)

        dfc = df.copy()

        # Check if the first row is a list or a string
        fr = df['features'].iloc[0]
        if isinstance(fr, str):
            # Convert from a string to a list
            dfc['features'] = df['features'].apply(ast.literal_eval)
        elif isinstance (fr, list):
            pass
        else:
            raise ValueError(f"Element is neither a list nor a string")

        for c in range(num_cat):
            # Extract the first element from each list into a new col
            dfc[c] = dfc['features'].apply(lambda x: x[c])

            # Get the number of unique values in that column
            unique_vals = dfc[c].unique()

            # For each unique value, create a one-hot code
            current_dict = {}

            for i in range(len(unique_vals)):
                current_dict[unique_vals[i]] = [0 if x != i else 1 for x in range(len(unique_vals))]
            
            # Convert the current category column into a one-hot encoding
            dfc[c] = dfc[c].apply(lambda x: current_dict[x])

        dfc['features'] = dfc['features'].apply(lambda x: x[c+1:])

        # Combine the one-hot encodings back into the feature column
        for c in reversed(indices):
            dfc['features'] = dfc[c] + dfc['features']

        # If dfc contains the column 'id', turn 'id' into the index
        if 'id' in dfc.columns:
            dfc.set_index('id', inplace=True)

        return dfc


    def scaleNum(self, df, indices):
        """ 
            Input: $df represents a DataFrame with two columns: "id" and "features"
                   $indices represents which dimensions in $features that need to be standardized
                    
            Output: Return a new DataFrame that updates "features" column with specified features standarized.
            
        """
        temp_df = pd.DataFrame(index=df.index)

        # Turn each value in each list of df['features'] into a column of temp_df
        for i in range(len(df['features'].iloc[0])):
            if i in indices:
                temp_df[i] = df['features'].apply(lambda x: x[i])
            else:
                temp_df[i] = df['features'].apply(lambda x: [x[i]])

        # Standardize the columns of temp_df by subtracting the mean and dividing by the standard deviation
        for i in indices:
            temp_df[i] = (temp_df[i] - temp_df[i].mean()) / temp_df[i].std()
        
        # Create col temp_df['features'] which combines each column of temp_df into a list
        # Create an empty column 'features' in temp_df
        temp_df['features'] = pd.Series([[]] * len(temp_df))

        for i in reversed(range(len(temp_df.columns)-1)):
            if i in indices:
                temp_df[i] = temp_df[i].apply(lambda x: [x])
            temp_df['features'] = temp_df[i] + temp_df['features']

        # Drop all cols except 'features'
        temp_df = temp_df[['features']]

        return temp_df

    def detect(self, df, k, t):
        """ 
            Input: $df represents a DataFrame with two columns: "id" and "features"
                $k is the number of clusters for K-Means
                $t is the score threshold
            
            Output: Return a new DataFrame that adds the "score" column into the input $df and then
                    removes the rows whose scores are smaller than $t.  
        """
        # Convert the 'features' column into a list of lists for KMeans
        X = df['features'].to_list()

        # Create a KMeans object and fit_predict the data
        kmeans = KMeans(n_clusters=k, random_state=None, n_init='auto').fit_predict(X)
        
        # Create a new DataFrame with the same indices as df and a new column 'cluster_index'
        df2 = pd.DataFrame(index=df.index)
        df2['cluster_index'] = kmeans
        
        # Join df and df2 on their indices
        df2 = df.join(df2)

        # Get the number of occurrences of each cluster
        cluster_counts = df2['cluster_index'].value_counts()
        Nmax = cluster_counts.max()
        Nmin = cluster_counts.min()
        
        # Compute the score for each record
        df2['score'] = df2['cluster_index'].apply(lambda x: (Nmax - cluster_counts[x]) / (Nmax - Nmin))

        df2 = df2[df2['score'] >= t].drop(columns=['cluster_index'])

        return df2

 
if __name__ == "__main__":
    
    # Runs on the sample data supplied
    df = pd.read_csv('logs-features-sample.csv').set_index('id')
    ad = AnomalyDetection()
    
    df1 = ad.cat2Num(df, [0,1])
    print(df1)

    df2 = ad.scaleNum(df1, [6])
    print(df2)

    df3 = ad.detect(df2, 8, 0.97)
    print(df3)

    # Runs on the test data supplied
    # data = [(0, ["http", "udt", 4]), \
    #     (1, ["http", "udf", 5]), \
    #     (2, ["http", "tcp", 5]), \
    #     (3, ["ftp", "icmp", 1]), \
    #     (4, ["http", "tcp", 4])]
    # df = pd.DataFrame(data=data, columns = ["id", "features"])    
    # ad = AnomalyDetection()
    
    # df1 = ad.cat2Num(df, [0,1])
    # print(df1)
    
    # df2 = ad.scaleNum(df1, [6])
    # print(df2)
    
    # df3 = ad.detect(df2, 2, 0.9)
    # print(df3)