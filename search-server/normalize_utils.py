import re

import numpy as np
from fuzzywuzzy import process

s1 = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ"
s0 = "AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy"


def remove_accents(input_str):
    s = ""
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None]*(n+1) for i in range(m+1)]

    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def abbreviate(s):
    return "".join(x[0] if not x[0].isnumeric() else x for x in s.split())


n2w = ["mot", "hai", "ba", "bon", "nam",
       "sau", "bay", "tam", "chin", "muoi"]
w2n = {w: i+1 for i, w in enumerate(n2w)}
wn_re = "|".join(n2w)


def normalize_district(district):
    district = remove_accents(district).lower().strip()
    if re.search(r"(q|quan|huyen|h)\W*[.,]?\W*\d+", district):
        return [str(int(re.search(r"\d+", district).group()))]
    if re.search(r"^(q|quan|huyen|h)\b", district):
        district = re.sub(r"\b(q|quan|huyen|h)\b", "", district, 1).strip()
        district = re.sub(r"\s*[.,]\s*", "", district).strip()
    value = [district]
    if len(district.split()) >= 2:
        value.append(abbreviate(district))
    return value


def normalize_ward(ward):
    ward = remove_accents(ward).lower().strip()
    if re.search(r"(p|phuong|xa|x)\W*[.,]?\W*\d+", ward):
        return [str(int(re.search(r"\d+", ward).group()))]
    if re.search(r"^(p|phuong|xa|x)\b", ward):
        ward = re.sub(r"\b(p|phuong|xa|x)\b", "", ward, 1).strip()
        ward = re.sub(r"\s*[.,]\s*", "", ward).strip()
    value = [ward]
    if len(ward.split()) >= 2:
        value.append(abbreviate(ward))
    return value


ROOMS = np.array([
    r"s(an)?\W*t(huong)?",
    r"san\b",
    r"(p(hong)?)?\W*t(ro)?|\btro\b",
    r"(p(hong)?)?\W*n(gu)?|\bngu\b",
    r"(p(hong)?)?\W*g(ia[tc])?|\bgia[tc]\b",
    r"(p(hong)?)?\W*t(ho)?\b|\btho\b",
    r"(p(hong)?)?\W*k(hach)?|\bkhach\b",
    r"n(ha)?\W*k(ho)?|\bkho\b",
    r"gara|o\W*to|xe\W*hoi",
    r"xe(\W*may)?",
    r"ki\W*o[ts]",
    r"(gieng\W*)?troi",
    r"van\W*phong|k(inh)?\W+d(oanh)?",
    r"ba[nl]g?\W*co(ng?|l)",
    r"(p(hong)?)?\W*(b(ep)?|\ban\b)|\bbep\W*an\b",
    r"(p(hong)?)?\W*(tam|v(e)?\W*s(inh)?|wc|toi?ll?e?t)",
    r"(p(hong)?)?\W+l(am)?\W+v(iec)?",
    r"(p(hong)?)?\W+s(inh)?\W+h(oat)?"
])
ROOM_NAME = np.array([
    "san thuong",
    "san",
    "phong tro",
    "phong ngu",
    "phong giat",
    "phong tho",
    "phong khach",
    "nha kho",
    "gara",
    "xe may",
    "kiots",
    "gieng troi",
    "van phong",
    "ban cong",
    "bep an",
    "nha ve sinh",
    "phong lam viec",
    "phong sinh hoat"
])
assert(len(ROOM_NAME) == len(ROOMS))
FLOORS = np.array([
    "tang", "lau", "tam", "me", "cap 4",
    "tang gac",
    "tang tret",
    "tang lung",
    "tang ham",
    "ban cong",
    "tang tum", "chuong cu",
    "san thuong",
])
FLOOR_NAME = np.array([
    "tang", "tang", "tang", "tang", "tang",
    "gac",
    "tret",
    "lung",
    "ham",
    "ban cong",
    "tum", "tum",
    "san thuong",
])
assert(len(FLOORS) == len(FLOOR_NAME))
ROOM_ABBRE = np.array([abbreviate(x) for x in ROOMS])
FLOOR_ABBRE = np.array([abbreviate(x) for x in FLOORS])
VOWEL = np.array(["a", "e", "i", "o", "u"])


def comma_split(word):
    return word.split(",")


# def analyze_part(part):
#     part = part.strip()
#     pattern = r"(\d+\s*-)+\d+"
#     if re.search(pattern, part):
#         return [{
#             "low": int(x.group()), "high": int(x.group())
#         } for x in re.finditer(r"\d+", part)]
#     pattern = r"\d+\s*(-|den|de|en|dn|toi|to|oi|ti)\s*\d+"
#     if re.search(pattern, part):
#         low, high = [int(x.group()) for x in re.finditer(r"\d+", part)]
#         return [{
#             "low": low,
#             "high": high
#         }]
#     if re.search(r"\d+", part):
#         return [{
#             "low": int(x.group()), "high": int(x.group())
#         } for x in re.finditer(r"\d+", part)]
num_pattern = np.array([
    "0",
    "0dentoi-0",
    "0-0-0"
])


def normalize_city(city):
    city = remove_accents(city).lower().strip()
    city = re.sub("\s*\.\s*", "", city)
    value = []
    a = ""
    for city in city.split('-'):  # For cases such as 'ba ria - vung tau'
        if city.startswith("thanh pho") or city.startswith("tp") or city.startswith("tinh"):
            city = re.sub(r"thanh\s*pho|tp|tinh", "", city, 1)
        city = city.strip()
        value.append(city)
        if len(city.split()) >= 2:
            b = abbreviate(city)
            a += b
            value.append(b)
    if a != '':
        value.append(a)
    return list(set(value))


def format_number_room_floor(s):
    pattern = r"(\d+\s*[,.]\s*)?\d+"
    start = 0
    l = []
    for match in re.finditer(pattern, s):
        l.append(s[start:match.start()].strip())
        l.append(re.sub(r"\s*[,.]\s*", ".", match.group()))
        start = match.end()
    l.append(s[start:])
    return " ".join(l)


def analyze_part(part):
    part = part.strip()
    part0 = re.sub(r"\s*(\d+([.]\d+)?)\s*", "0", part)
    part0 = re.match(r"0(.*0)?", part0)
    if part0 is None:
        return [{"low": 1.0, "high": 1.0}]
    part0 = part0.group()
    score = np.array([lcs(part0, x) for x in num_pattern])
    i = np.argmax(score)
    if i == 0 or i == 2:
        return [{
            "low": float(x.group()), "high": float(x.group())
        } for x in re.finditer(r"\d+([.]\d+)?", part)]
    elif i == 1:
        low, high = (x.group() for x in re.finditer(r"\d+([.]\d+)?", part))
        return [{"low": float(low), "high": float(high)}]


def normalize_room(room):
    """
        Cần phải xử lý các trường hợp sau:
        - 1 - 3 phòng ngủ
        - 1 - 3 - 5 phòng ngủ
        - 1, 3, 5 phòng ngủ
        - 1 tới 3 phòng ngủ
        - 1 đến 3 phòng ngủ
        - 3 phòng ngủ
        - Các trường hợp trên nhưng số biểu diễn bằng chữ.
    """
    try:
        room = remove_accents(room).lower()
        room = room.split()
        temp = ' '.join(room[:-1])
        room = room[-1]
        for w, i in w2n.items():
            temp = re.sub(r"\b{}\b".format(w), "{}".format(i), temp)
        room = ' '.join([temp, room])
        room = format_number_room_floor(room)
        values = analyze_part(room)
        pattern = r"(\d+([^\w,]+\d+)+|\d+|,)"
        room = re.sub(pattern, "", room).strip()
        room = re.sub(r"\s+", 'SPACE', room)
        room = re.sub(r"\W+", "", room).strip()
        room = re.sub('SPACE', ' ', room).strip()
        score = []
        for p in ROOMS:
            m = re.search(p, room)
            if m is None:
                score.append(0)
            else:
                score.append(max([len(x.group())
                                  for x in re.finditer(p, room)]))
        # print(room, score)
        return {
            ROOM_NAME[np.argmax(score)]: values
        } if np.max(score) >= 2 else {
            'phong': values
        } if room == 'p' or room == 'phong' or room == 'ph' else {}
    except:
        return {}


def normalize_floor(floor):
    """
        Cần phải xử lý các trường hợp sau:
        - 1 - 3 phòng ngủ
        - 1 - 3 - 5 phòng ngủ
        - 1, 3, 5 phòng ngủ
        - 1 tới 3 phòng ngủ
        - 1 đến 3 phòng ngủ
        - 3 phòng ngủ
        - Các trường hợp trên nhưng số biểu diễn bằng chữ.
    """
    try:
        floor = remove_accents(floor).lower()
        floor = floor.split()
        if floor[-1] in FLOORS:
            temp = ' '.join(floor[:-1])
            floor = floor[-1]
        elif len(floor) > 2 and floor[-2] in FLOORS and floor[-1] == 'ruoi':
            temp = ' '.join(floor[:-2])
            floor = ' '.join(floor[-2:])
        else:
            temp = ' '.join(floor)
            floor = ''
        for w, i in w2n.items():
            temp = re.sub(r"\b{}\b".format(w), "{}".format(i), temp)
        floor = ' '.join((temp, floor))
        floor = format_number_room_floor(floor)
        values = analyze_part(floor)
        pattern = r"(\d+([^\d,]+\d+)+|\d+|,)"
        floor = re.sub(pattern, "", floor).strip()
        floor = re.sub(r"\s+", "SPACE", floor)
        floor = re.sub(r"\W+", "", floor).strip()
        floor = floor.replace("SPACE", " ")
        if re.search(r"\bruoi\b", floor):
            values = [{"low": v["low"]+0.5, "high":v["high"]+0.5}
                      for v in values]
            floor = re.sub(r"\bruoi\b", "", floor)
        floor = floor.replace(" ", "")
        score_f = np.array([lcs(floor, x) for x in FLOORS])
        i_f = np.argmax(score_f)
        score_a = np.array([lcs(floor, x) for x in FLOOR_ABBRE])
        i_a = np.argmax(score_a)
        if score_a[i_a] < score_f[i_f]:
            return {
                FLOOR_NAME[i_f]: values
            }
        else:
            return {
                FLOOR_NAME[i_a]: values
            }
    except:
        return {}


def format_number_area(s):
    pattern = r"(\d+\s*[,.]\s*)?\d+"
    start = 0
    l = []
    for match in re.finditer(pattern, s):
        l.append(s[start:match.start()].strip())
        num = re.sub(r"\s*[,.]\s*", ".", match.group())
        di = num.find(".")
        t = 1 if di < 0 else len(num) - di
        if t > 3:
            num = num.replace(".", "")
        l.append(num)
        start = match.end()
    l.append(s[start:])
    return " ".join(l)


def normalize_area(area):
    """
    Các trường hợp cần xử lý:
    - 100 m 2
    - 100 m 2 = 20 m x 20 m và 20 m x 20 m = 100 m 2
    - 100 - 200 m 2 và 100 m 2 - 200 m 2
    - hectar
    """
    try:
        area = area.lower().strip()
        area = remove_accents(area)
        area = re.sub(r"\b(den|de|en|dn|toi|to|oi|ti)\b", "-", area)
        if re.search(r"(\d+(\s*[.]\s*\d+)?\s*[x*]\s*\d+(\s*[.]\s*\d+)?\s*,\s*)+(\d+(\s*[.]\s*\d+)?\s*[x*]\s*\d+(\s*[.]\s*\d+)?)", area):
            default_metric = "m 2" if re.search(
                r"m\s*\d", area) else "ha" if "ha" in area or "hec" in area else "m 2"
            area = [format_number_area(x.group()) for x in re.finditer(
                r"\d+(\s*[.]\s*\d+)?\s*[x*]\s*\d+(\s*[.]\s*\d+)?", area)]
        elif re.search(r"(\d+(\s*[.]\s*\d+)?\s*,\s*){2,}\d+(\s*[.]\s*\d+)?", area):
            default_metric = "m 2" if re.search(
                r"m\s*\d", area) else "ha" if "ha" in area or "hec" in area else "m 2"
            area = [format_number_area(x.strip()) for x in area.split(',')]
        else:
            if re.search("m\s*2\s*[.,]\s*\d+\s*[*x]\s*\d+", area):
                area = re.sub(r"m\s*2\s*[.,]", "m 2 =", area)
            elif re.search("m\s*2\s*[.,]\s*\d+", area):
                area = re.sub(r"m\s*2\s*[.,]", "m 2 -", area)
            area = format_number_area(area)
            default_metric = "m 2" if re.search(
                r"m\s*\d", area) else "ha" if "ha" in area or "hec" in area else "m 2"
            area = [x for x in area.split(
                "-") if x != "" and not x.isspace() and re.search(r"\d+", x)]
        values = []
        for part in area:
            sm = [x for x in part.split(
                "=") if x != "" and not x.isspace() and re.search(r"\d+", x)]
            value = {}
            if len(sm) == 2:
                a, b = sm
                ma = "m 2" if re.search(
                    r"m\s*\d+", a) else "ha" if "ha" in a or "hec" in a else default_metric
                mb = "m 2" if re.search(
                    r"m\s*\d+", b) else "ha" if "ha" in b or "hec" in b else default_metric
                a = re.sub(r"m\s*\d+", "", a)
                b = re.sub(r"m\s*\d+", "", b)
                va = [float(x.group())
                      for x in re.finditer(r"\d+(\s*[,.]\s*\d+)?", a)]
                vb = [float(x.group())
                      for x in re.finditer(r"\d+(\s*[,.]\s*\d+)?", b)]
                if len(va) == 2:
                    value["dai"], value["rong"] = max(va), min(va)
                    if ma == "ha":
                        value["dai"] *= 100
                        value["rong"] *= 100
                    value["dien tich"] = vb[0]
                    if mb == "ha":
                        value["dien tich"] *= 10000
                elif len(vb) == 2:
                    value["dai"], value["rong"] = max(vb), min(vb)
                    if mb == "ha":
                        value["dai"] *= 100
                        value["rong"] *= 100
                    value["dien tich"] = va[0]
                    if ma == "ha":
                        value["dien tich"] *= 10000
            else:
                a = sm[0]
                ma = "m 2" if re.search(
                    r"m\s*\d+", a) else "ha" if "ha" in a or "hec" in a else default_metric
                a = re.sub(r"m\s*\d+", "", a)
                va = [float(x.group())
                      for x in re.finditer(r"\d+(\s*[,.]\s*\d+)?", a)]
                if len(va) == 2:
                    value["dai"], value["rong"] = max(va), min(va)
                    if ma == "ha":
                        value["dai"] *= 100
                        value["rong"] *= 100
                    value["dien tich"] = value["dai"]*value["rong"]
                else:
                    value["dien tich"] = va[0]
                    if ma == "ha":
                        value["dien tich"] *= 10000
            values.append(value)
        if len(values) == 2:
            return [
                {"low": values[0],
                 "high": values[1]}
            ]
        return [
            {
                "low": x,
                "high": x
            } for x in values
        ]
    except:
        return []


ORIENTATION = np.array(['dong', 'nam', 'tay', 'bac', 'dong bac',
                        'dong nam', 'tay bac', 'tay nam', 'db', 'dn', 'tb', 'tn'])


def normalize_orientation(orientation):
    s = remove_accents(orientation).lower().strip()
    res, score = process.extractOne(s, ORIENTATION)
    if score < 80:
        return []
    if len(res.split()) >= 2:
        return [res, abbreviate(res)]
    return [res]


def analyze_price(price, default):
    num = re.search(r"\d+(\s*[,.]\s*\d+)*", price).group()
    word = re.sub(num, '', price)
    temp = re.sub(r"[\s.,]", "", num)
    if len(temp) >= 6:
        return float(temp)/1e6
    num = re.sub(r"\s*[,.]\s*", ".", num)
    num = re.search(r"\d+(\s*[.]\s*\d+)?", num).group()
    if 'ty' in word or 'ty' in default:
        return float(num)*1e3
    if 'usd' in word or 'usd' in default:
        return float(num)*22770/1e6
    if 'nghin' in word or 'ngan' in word or re.search(r"\bk\b", word) or default == 'k':
        return float(num)/1e3
    return float(num)


def normalize_price(price):
    try:
        price = remove_accents(price).lower().replace('ti', 'ty').strip()
        price = re.sub(r"\s*1 th(ang)?|\d*\s*m\s*\d+|\d+\s*m\s*\d*",
                       "", price).strip()
        price = re.sub(r"\$", "usd", price)
        price = re.sub(r"(-\s*)*-", "-", price)
        for x in re.finditer(r"\d+\s*(ty|trieu|tr)\s*\d+(\s*[.,]\s*\d+)?", price):
            x = x.group()
            a, b = re.finditer(r"\d+(\s*[.,]\s*\d+)?", x)
            replace = ' '.join([a.group(), '.', re.sub(
                r"[.,]", "", b.group()), re.sub(r"\d+(\s*[.,]\s*\d+)?", "", x)])
            price = re.sub(x, replace, price)
        pattern = r"(\d+(\s*[,.]\s*\d+)*\s*(ty|tr|trieu|usd)?[^\d]*(-|toi|den)\s*)*\d+(\s*[,.]\s*\d+)*\s*(ty|tr|trieu|usd)?"
        parts = [x.group().strip() for x in re.finditer(pattern, price)]
        default_metric = 'ty' if 'ty' in price else 'usd' if 'usd' in price else 'k' if 'nghin' in price or 'ngan' in price or ' k ' in price else 'tr'
        values = []
        for part in parts:
            nums = re.split(r"-|toi|den", part)
            if len(nums) == 2:
                a, b = nums
                values.append({'low': analyze_price(a, default_metric),
                               'high': analyze_price(b, default_metric)})
            else:
                nums = (analyze_price(x, default_metric) for x in nums)
                values.extend({
                    'low': x, 'high': x
                } for x in nums)
        return values
    except:
        return []


TRANSACTION = np.array(
    ['mua', 'ban', 'cho thue', 'can thue', 'sang nhuong', 'can tim'])
TRANSACTION_NAME = np.array(
    ['mua', 'ban', 'cho thue', 'can thue', 'sang nhuong', 'can tim'])
TRANSACTION_INDEX = {w: i for i, w in enumerate(TRANSACTION)}
assert(len(TRANSACTION) == len(TRANSACTION_NAME))


def normalize_transaction(trans):
    trans = remove_accents(trans).lower().strip()
    res, score = process.extractOne(trans, TRANSACTION)
    if score < 50:
        return []
    return [TRANSACTION_NAME[TRANSACTION_INDEX[res]]]


POSITION = np.array(['hem', 'hxh', 'ngo', 'mat tien',
                     'mat pho', 'mat duong', 'mt', 'mp', 'md'])
POSITION_NAME = np.array(['hem', 'hem', 'hem', 'mat tien',
                          'mat tien', 'mat tien', 'mat tien', 'mat tien', 'mat tien'])
POSITION_INDEX = {w: i for i, w in enumerate(POSITION)}
assert(len(POSITION) == len(POSITION_NAME))


def normalize_position(pos):
    pos = remove_accents(pos).lower().strip()
    res, score = process.extractOne(pos, POSITION)
    if score < 80:
        return []
    return [POSITION_NAME[POSITION_INDEX[res]]]


LEGAL = np.array(['so do', 'so hong', 'sd', 'sh',
                  'giay phep xay dung', 'gpxd'])
LEGAL_NAME = np.array(
    ['so hong do', 'so hong do', 'so hong do', 'so hong do', 'gpxd', 'gpxd'])
LEGAL_INDEX = {w: i for i, w in enumerate(LEGAL)}
assert(len(LEGAL) == len(LEGAL_NAME))


def normalize_legal(legal):
    legal = remove_accents(legal).lower().strip()
    res, score = process.extractOne(legal, LEGAL)
    if score < 50:
        return []
    return [LEGAL_NAME[LEGAL_INDEX[res]]]


REALESTATE = np.array(['nha', 'dat', 'can ho', 'chung cu', 'biet thu', 'villa', 'phong tro', 'nha tro', 'phong',
                       'cua hang', 'shop', 'kiots', 'quan', 'khach san', 'xuong', 'nha xuong', 'kho', 'van phong', 'mat bang', 'toa nha'])
REALESTATE_NAME = np.array(['nha', 'dat', 'can ho', 'can ho', 'biet thu', 'biet thu', 'tro', 'tro', 'tro', 'cua hang',
                            'cua hang', 'kiot', 'cua hang', 'khach san', 'xuong', 'xuong', 'kho', 'van phong', 'mat bang', 'toa nha'])
REALESTATE_INDEX = {w: i for i, w in enumerate(REALESTATE)}
assert(len(REALESTATE) == len(REALESTATE_NAME))


def normalize_realestate(re):
    re = remove_accents(re).lower().strip()
    res, score = process.extractOne(re, REALESTATE)
    if score < 70:
        return []
    if len(res.split()) >= 2:
        return [REALESTATE_NAME[REALESTATE_INDEX[res]], abbreviate(res)]
    return [REALESTATE_NAME[REALESTATE_INDEX[res]]]


FUNCTIONS = {
    'addr_city': normalize_city,
    'addr_ward': normalize_ward,
    'addr_district': normalize_district,
    'orientation': normalize_orientation,
    'legal': normalize_legal,
    'position': normalize_position,
    'transaction_type': normalize_transaction,
    'realestate_type': normalize_realestate,
    'price': normalize_price,
    'interior_floor': normalize_floor,
    'interior_room': normalize_room,
    'area': normalize_area
}

if __name__ == "__main__":
    print(normalize_room("3.3 ve "))
    print(normalize_room("3.3 phong tam "))
    print(normalize_floor("st"))
    print(normalize_floor("1 gác lững ruoi"))
    print(normalize_floor("ban cong"))
    print(normalize_area("ngang 4 x 6"))
    print(lcs("gaclung", "chuong cu"))
    print(normalize_ward('p ,12'))
    print(normalize_area('75 , 120 , 202'))
    print(normalize_room('ki - ot'))
    print(normalize_area("10 m 2 - 200 m 2"))
    print(normalize_city("tp ho chi minh"))
    print(normalize_realestate("căn hộ"))
    print(normalize_price('1.3 ti - 2,5 ty'))
