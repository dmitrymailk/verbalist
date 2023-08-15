### удалить большой файл из истории гита

```bash
git filter-branch -f --tree-filter 'rm -f verbalist/datasets/openchat_sharegpt4_dataset/openchat_sharegpt4_dataset.ipynb' HEAD
```
