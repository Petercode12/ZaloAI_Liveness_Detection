from utils import file2class
import tqdm, os
import pandas as pd

nameCol = []
resCol = []
for filename in tqdm.tqdm(os.listdir('./public_test_2/public_test_2/videos/')):

    result = file2class(f'./public_test_2/public_test_2/videos/{filename}')
    nameCol.append(filename)
    resCol.append(result)

df = pd.DataFrame({
    'fname': nameCol,
    'liveness_score': resCol
})

print(df)

df.to_csv('predict.CSV', index = False)
print('DONE')