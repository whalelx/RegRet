1. make_qa_data 
to generate all jsons and boxed imgs
2. gen_stage2_data 
to pick questions from these jsons and merge them into one sharegpt-format json
3. merge_final 
to combine all 3 data splits
4. ../replace-datapath.sh 