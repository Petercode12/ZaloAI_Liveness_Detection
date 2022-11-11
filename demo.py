from utils import file2class
import tqdm, os
import pandas as pd

result = file2class('./public/videos/100.mp4')
print(result)

nameCol = []
resCol = []
for filename in tqdm.tqdm(os.listdir('public/videos')):

    result = file2class(f'./public/videos/{filename}')
    nameCol.append(filename)
    resCol.append(result)

df = pd.DataFrame({
    'fname': nameCol,
    'liveness_score': resCol
})

print(df)

df.to_csv('predict.CSV', index = False)
print('DONE')