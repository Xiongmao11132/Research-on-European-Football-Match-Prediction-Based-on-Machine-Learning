import re

def process_ipv6_addresses(input_file, output_file):
    # 从输入文件读取内容
    with open(input_file, 'r') as infile:
        input_text = infile.read()

    # 正则表达式匹配IPv6地址
    ipv6_pattern = r'2001:[0-9a-fA-F:]+/64'

    # 提取所有IPv6地址
    ipv6_addresses = re.findall(ipv6_pattern, input_text)

    # 将每个IPv6地址换行排列
    output_text = "\n".join(ipv6_addresses)

    # 写入输出文件
    with open(output_file, 'w') as outfile:
        outfile.write(output_text)

# 示例调用
input_file = r"D:\desk\coinads国外广告平台项目\ip.txt"  # 输入文件路径
output_file = r"D:\desk\coinads国外广告平台项目\t.txt"  # 输出文件路径
process_ipv6_addresses(input_file, output_file)