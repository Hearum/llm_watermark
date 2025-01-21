import pstats

# 加载 .prof 文件
p = pstats.Stats('/home/shenhm/documents/output.prof')

# 打印简单的统计信息
p.strip_dirs()  # 去掉路径中的多余部分，便于查看
p.sort_stats('cumtime').print_stats(100)
