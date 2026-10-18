def extract_cpu_brand(text):
    text_list = text.split()

    # Lấy 3 từ đầu tiên (ví dụ: "Intel Core i5")
    first_3_words = ' '.join(text_list[:3])

    if first_3_words in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
        return first_3_words
    elif text_list[0] == 'Intel':
        return 'Other Intel Processor'  # Celeron, Pentium, Core M, Xeon...
    elif text_list[0] == 'AMD':
        return 'AMD Processor'  # A-Series, E-Series, Ryzen...
    else:
        return 'Other'  # Samsung Cortex...

def extract_cap(series, tech_name):
    # Kiểm tra xem chuỗi có chứa loại ổ cứng đó không
    has_tech = series.str.contains(tech_name, na=False)
    # Trích xuất phần số đứng trước tên loại ổ cứng
    cap = series.str.extract(r'(\d+)\s*' + tech_name)[0].fillna(0).astype(int)
    return cap * has_tech

def process_gpu_type(text):
    if 'Nvidia' in text:
        if 'Quadro' in text:
            return 'Nvidia Quadro'
        return 'Nvidia GeForce'
    elif 'AMD' in text:
        if 'FirePro' in text:
            return 'AMD FirePro'
        return 'AMD Radeon'
    elif 'Intel' in text:
        return 'Intel Integrated Graphics'
    else:
        return 'Other'