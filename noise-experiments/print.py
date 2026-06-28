import re
import sys

def parse_file(filename):
    # mode1
    # wanted_recall_idx = []
    # expected_groups = [
    #     ('coco0', re.compile(r'mbeir_mscoco_task0')),
    #     ('coco3', re.compile(r'mbeir_mscoco_task3')),
    #     ('cirr7', re.compile(r'mbeir_cirr_task7')),
    #     ('webqa1', re.compile(r'mbeir_webqa_task1')),
    #     ('fiq7', re.compile(r'mbeir_fashioniq_task7')),
    #     ('nights4', re.compile(r'mbeir_nights_task4')),
    #     ('fashion200k_0', re.compile(r'mbeir_fashion200k_task0')),
    #     ('visualnews_task0', re.compile(r'mbeir_visualnews_task0')),
    #     ('webqa_task2', re.compile(r'mbeir_webqa_task2')),
    #     ('fashion200k_3', re.compile(r'mbeir_fashion200k_task3')),
    #     ('visualnews_task3', re.compile(r'mbeir_visualnews_task3')),
    #     ('edis_task2', re.compile(r'mbeir_edis_task2'))
    # ]

    # mode2
    # wanted_recall_idx = [ 1,1,2, 1,1,1, 1,1,2,1,1,1,2,1,1,1 ]
    # wanted_recall_idx = [ 1,2 ,1,1, 1,2,1,1,2,1,1,1 ]
    # expected_groups = [
    #     # ('visualnews_task0', re.compile(r'mbeir_visualnews_task0')),
    #     ('coco0', re.compile(r'mbeir_mscoco_task0')),
    #     ('fashion200k_0', re.compile(r'mbeir_fashion200k_task0')),
    #     # ('webqa1', re.compile(r'mbeir_webqa_task1')),
    #     ('edis_task2', re.compile(r'mbeir_edis_task2')),
    #     ('webqa_task2', re.compile(r'mbeir_webqa_task2')),
    #     # ('visualnews_task3', re.compile(r'mbeir_visualnews_task3')),
    #     ('coco3', re.compile(r'mbeir_mscoco_task3')),
    #     ('fashion200k_3', re.compile(r'mbeir_fashion200k_task3')),
    #     ('nights4', re.compile(r'mbeir_nights_task4')),
    #     # ('oven6', re.compile(r'mbeir_oven_task6')),
    #     ('infoseek6', re.compile(r'mbeir_infoseek_task6')),
    #     ('fiq7', re.compile(r'mbeir_fashioniq_task7')),
    #     ('cirr7', re.compile(r'mbeir_cirr_task7')),
    #     ('oven8', re.compile(r'mbeir_oven_task8')),
    #     ('infoseek8', re.compile(r'mbeir_infoseek_task8')),
    # ]

    # mode3
    # wanted_recall_idx = []
    # expected_groups = [
    #     ('dam0', re.compile(r'mbeir_dam_task0')),
    #     ('dam3', re.compile(r'mbeir_dam_task3')),
    #     ('fgclip3', re.compile(r'mbeir_fgclip_task3')),
    #     ('xhsnote4', re.compile(r'mbeir_xhsnote_task4')),
    #     ('imgdiff6', re.compile(r'mbeir_imgdiff_task6')),
    #     ('vismin6', re.compile(r'mbeir_vismin_task6')),
    # ]

    # mode4
    wanted_recall_idx = [1,1,1,1,1,1,0,0,1]
    expected_groups = [
        ('dam0', re.compile(r'mbeir_dam_task0')),
        ('fgclip0', re.compile(r'mbeir_fgclip_task0')),
        ('dam3', re.compile(r'mbeir_dam_task3')),
        ('fgclip3', re.compile(r'mbeir_fgclip_task3')),
        ('xhsnote4', re.compile(r'mbeir_xhsnote_task4')),
        ('xhsgood4', re.compile(r'mbeir_xhsgood_task4')),
        ('vismin6', re.compile(r'mbeir_vismin_task6')),
        ('imgdiff6', re.compile(r'mbeir_imgdiff_task6')),
        # ('xhsgood7', re.compile(r'mbeir_xhsgoodmoreneg_task7_merge')),
        ('xhsgood7', re.compile(r'mbeir_xhsgood_task7_merge')),
    ]

    
    # 初始化结果字典
    results = {group_name: [0, 0, 0, 0] for group_name, _ in expected_groups}
    
    with open(filename, 'r') as file:
        current_group = None
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是路径行（新组开始）
            if line.startswith('/'):
                current_group = None
                for group_name, pattern in expected_groups:
                    if pattern.search(line):
                        current_group = group_name
                        break
            elif current_group and line.startswith('recall_at_'):
                # 解析指标值
                try:
                    parts = line.split('=')
                    if len(parts) == 2:
                        metric = parts[0].strip()
                        value = float(parts[1].strip())
                        
                        if 'recall_at_10' in metric:
                            results[current_group][2] = value
                        elif 'recall_at_50' in metric:
                            results[current_group][3] = value
                        elif 'recall_at_1' in metric:
                            results[current_group][0] = value
                        elif 'recall_at_5' in metric:
                            results[current_group][1] = value
                except:
                    continue
    
    # 按预期顺序输出结果，每个数字单独一行
    output = []
    if wanted_recall_idx is not []:
        for group_name, _ in expected_groups:
            metrics = results.get(group_name, [0, 0, 0, 0])
            idx = wanted_recall_idx.pop(0)
            metrics = metrics[idx:idx+1]
            for metric in metrics:
                # Convert to percentage and round to 3 decimal places
                percentage = round(metric * 100, 3)
                # Format the string to remove trailing zeros after 3 decimal places
                metric_str = "{0:.1f}".format(percentage)
                if metric_str.endswith(".000"):
                    metric_str = metric_str[:-4]
                output.append(metric_str)
        return ' & '.join(output)

    else:
        for group_name, _ in expected_groups:
            metrics = results.get(group_name, [0, 0, 0, 0])
            for metric in metrics:
                # 将浮点数转为字符串，去掉末尾的零和小数点（如果是整数）
                metric_str = f"{metric:.6f}"
                if '.' in metric_str:
                    metric_str = metric_str.rstrip('0').rstrip('.')
                output.append(metric_str)
    
        return '\n'.join(output)

def calc_group_means(data_str):
    # data_str = "55.2 & 72.7 & 72.9 & 85.5 & 86.5 & 91.5 & 81.7 & 85.5 & 96.2"
    # 将字符串按 & 分割并转成浮点数
    nums = [float(x) for x in data_str.split("&")]
    
    # 定义分组（使用下标，从1开始）
    groups = [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8, 9)
    ]
    
    # 计算各组平均值
    group_means = []
    for g in groups:
        group_values = [nums[i - 1] for i in g]  # i-1 因为Python下标从0开始
        mean_val = sum(group_values) / len(group_values)
        group_means.append(mean_val)
    
    # 计算总平均值
    total_mean = sum(nums) / len(nums)


    print("group means: ", end='')
    for a in group_means:
        print(f"{a:.1f}", end=' & ')
    
    print(f" {total_mean:.1f}")
    # print(f"\ntotal mean: {total_mean:.1f}")

    return group_means, total_mean


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
        
    filename = sys.argv[1]
    try:
        ans_str = parse_file(filename)
        # print(ans_str)
        print(filename)
        means, total_mean = calc_group_means(ans_str)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
