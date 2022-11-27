from utils import file2class
import tqdm, os
import pandas as pd

nameCol = []
resCol = []
for filename in tqdm.tqdm(os.listdir('data')):
    result = file2class(f'./data/{filename}')
    nameCol.append(filename)
    resCol.append(result)

df = pd.DataFrame({
    'fname': nameCol,
    'liveness_score': resCol
})

print(df)

if not os.path.exists('result'): 
    os.mkdir('result')
df.to_csv('./result/submission.CSV', index = False)
print('DONE')