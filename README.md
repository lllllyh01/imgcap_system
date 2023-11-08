The code of Attention Enhanced Network with Semantic Inspector is in directory /model_vscode.
Train model:
1. Modify `model` in /model_vscode/train_vit_selfatt_lbpf_.py to the model name you like (make sure it is the same when evaluating model)
2. run train_vit_selfatt_lbpf_.py
   ```
   python /model_vscode/train_vit_selfatt_lbpf_.py
   ```

Test model:
run eval_vit_selfatt_lbpf_.py
```
python /model_vscode/eval_vit_selfatt_lbpf_.py `model`(the model name you defined in train)
```

TODO: Upload dataset
