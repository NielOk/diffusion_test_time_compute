read -p "search_method: " search_method
read -p "approach: " approach

nohup python -u exp1_caching_fix.py  --approach $approach --search_method $search_method > approach_${approach}_search_method_${search_method}.log 2>&1 &