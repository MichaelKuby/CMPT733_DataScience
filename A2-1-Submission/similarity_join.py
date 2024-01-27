# similarity_join.py
import ast
import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def preprocess_df(self, df, cols):
        """
            Input:  df represents a pandas DataFrame
                    cols represents the list of columns in df that will be concatenated and tokenized.

            Output: Return a new DataFrame that adds the "joinKey" column to the input df
        """

        def split_and_concat(row):

            # Create one large string from the concatenation of each specified cols

            concatenated = ' '.join([str(row[col]) for col in cols if row[col] is not None])

            # Split the string into tokens, removing puncutation, and convert to lowercase

            tokens = [token.lower() for token in re.split(r'\W+', concatenated) if token and token.lower() != 'nan']
            return str(tokens)

        # Create a new df and apply the split_and_concat function to each row

        new_df = df.copy()
        new_df['joinKey'] = new_df.apply(split_and_concat, axis=1)

        return new_df

    def filtering(self, df1, df2):
        """
            Input: $df1 and $df2 are two input DataFrames, where each of them
                has a 'joinKey' column added by the preprocess_df function

            Output: Return a new DataFrame $cand_df with four columns: 'id1', 'joinKey1', 'id2', 'joinKey2',
                    where 'id1' and 'joinKey1' are from $df1, and 'id2' and 'joinKey2'are from $df2.
                    Intuitively, $cand_df is the joined result between $df1 and $df2 on the condition that
                    their joinKeys share at least one token.

            Comments: Since the goal of the "filtering" function is to avoid n^2 pair comparisons,
                    you are NOT allowed to compute a cartesian join between $df1 and $df2 in the function.
                    Please come up with a more efficient algorithm (see hints in Lecture 2).
        """

        # Convert all 'joinKey' strings to lists

        df1['joinKey'] = df1['joinKey'].apply(ast.literal_eval)
        df2['joinKey'] = df2['joinKey'].apply(ast.literal_eval)

        # Step 1: Build inverse indices for the second dataframe

        def build_token_index(df):
            token_index = {}

            # Iterate over the rows of the frst dataframe

            for index, row in df.iterrows():

                # Get the row id associated with the token and add it to the set

                for token in row['joinKey']:
                    if token not in token_index:
                        token_index[token] = set()
                    token_index[token].add(row['id'])
            return token_index

        token_index_df2 = build_token_index(df2)

        # Step 2: Build a list of matches

        matches = []

        for index, row in df1.iterrows():

            # Build a set containing all of the ids in token_index_df2 that share a token with the current row

            matched_ids = set()
            for token in row['joinKey']:
                matched_ids.update(token_index_df2.get(token, []))

            # For each id, make a dict and add to the list of matches

            for id2 in matched_ids:
                new_tuple = {'id1': row['id'],
                            'joinKey1': row['joinKey'],
                            'id2': id2,
                            'joinKey2': df2[df2['id'] == id2]['joinKey'].iloc[0]}
                matches.append(new_tuple)

        # 'joinKey' columns is a list, not a string, so convert to string

        df1['joinKey'] = df1['joinKey'].apply(lambda x: str(x))
        df2['joinKey'] = df2['joinKey'].apply(lambda x: str(x))


        # Step 3: Build the resultant dataframe

        cand_df = pd.DataFrame(matches)

        # Convert all 'joinKey' lists back to strings

        cand_df['joinKey1'] = cand_df['joinKey1'].apply(lambda x: str(x))
        cand_df['joinKey2'] = cand_df['joinKey2'].apply(lambda x: str(x))

        return cand_df


    def verification(self, cand_df, threshold):
        """
        Input: $cand_df is the output DataFrame from the 'filtering' function.
               $threshold is a float value between (0, 1]

        Output: Return a new DataFrame $result_df that represents the ER result.
                It has five columns: id1, joinKey1, id2, joinKey2, jaccard

        Comments: There are two differences between $cand_df and $result_df
                  (1) $result_df adds a new column, called jaccard, which stores the jaccard similarity
                      between $joinKey1 and $joinKey2
                  (2) $result_df removes the rows whose jaccard similarity is smaller than $threshold
        """

        # Convert both columns to sets for easier computation

        cand_df['joinKey1'] = cand_df['joinKey1'].apply(lambda x: set(ast.literal_eval(x)))
        cand_df['joinKey2'] = cand_df['joinKey2'].apply(lambda x: set(ast.literal_eval(x)))

        # Compute the Jaccard similarity for each pair,
        # computed as the length of their intersection divided by the length of their union

        cand_df['jaccard'] = cand_df.apply(lambda x:
            len(x['joinKey1'].intersection(x['joinKey2'])) /
            len(x['joinKey1'].union(x['joinKey2'])), axis=1)

        # Remove the rows whose Jaccard similarity is smaller than the threshold

        result_df = cand_df[cand_df['jaccard'] >= threshold].copy()

        # Convert both columns back to strings

        result_df['joinKey1'] = result_df['joinKey1'].apply(lambda x: str(x))
        result_df['joinKey2'] = result_df['joinKey2'].apply(lambda x: str(x))

        return result_df


    def evaluate(self, result, ground_truth):
        """
            Input: $result is a list of matching pairs identified by the ER algorithm
                $ground_truth is a list of matching pairs labeld by humans

            Output: Compute precision, recall, and fmeasure of $result based on $ground_truth, and
                    return the evaluation result as a triple: (precision, recall, fmeasure)

        """

        # Calculate T, the number of found results also appearing in the ground truth
        num = 0

        for pair in result:
            if pair in ground_truth:
                num += 1

        T = num
        R = len(result)
        A = len(ground_truth)

        # Calculate precision, |T| / |R| where R is the results found by our algorithm
        precision = T / R

        # Calculate recall, |T| / |A| where A is the labelled ground truth
        recall = T / A

        # Calculate fmeasure
        fmeasure = (2 * precision * recall) / (precision + recall)

        return (precision, recall, fmeasure)

    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0]))

        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))

        return result_df



if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))

"""The program will output the following when running on the sample data:


> Before filtering: 256 pairs in total

> After Filtering: 84 pairs left

> After Verification: 6 similar pairs

> (precision, recall, fmeasure) =  (1.0, 0.375, 0.5454545454545454)
"""