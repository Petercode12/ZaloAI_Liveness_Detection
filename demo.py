from utils import file2class
import tqdm, os
import pandas as pd

# result = file2class('./public/videos/100.mp4')
result = file2class('./public_test_2/videos/100.mp4')
print(result)

nameCol = []
resCol = []
# for filename in tqdm.tqdm(os.listdir('public/videos')):
for filename in tqdm.tqdm(os.listdir('public_test_2/videos')):
    # result = file2class(f'./public/videos/{filename}')
    result = file2class(f'./public_test_2/videos/{filename}')
    nameCol.append(filename)
    resCol.append(result)

df = pd.DataFrame({
    'fname': nameCol,
    'liveness_score': resCol
})

print(df)

df.to_csv('./result/submission.CSV', index = False)
print('DONE')