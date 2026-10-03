"""Thai -> English display names for products, categories and channels (display only; IDs/SKUs are never changed).
Names not found here are shown unchanged, so uploaded datasets with other products still work."""
import re

FORMS = {'ร้อน': 'Hot', 'เย็น': 'Iced', 'ปั่น': 'Blended'}

CATEGORIES = {
    'COCOA&MILK': 'Cocoa & Milk', 'COFFEE': 'Coffee', 'ITALIAN SODA': 'Italian Soda', 'SMOOTHIE': 'Smoothie', 'TEA': 'Tea',
    'Topping ต่างๆ': 'Toppings', 'ขนมขบเคี้ยว': 'Snacks', 'ขนมของฝาก/อื่นๆ': 'Souvenir Snacks / Other', 'ของที่ระลึก': 'Souvenirs',
    'น้ำอัดลม': 'Soft Drinks', 'มาม่า/โจ็ก': 'Instant Noodles / Congee', 'อาหาร': 'Food', 'อื่นๆ': 'Other',
    'เครื่องดื่ม/อาหารเจ': 'Vegetarian Food & Drinks', 'เบเกอรี่': 'Bakery',
}

CHANNELS = {
    'หน้าร้าน (Dine-In)': 'In-store (Dine-in)', 'เว็บไซต์ร้าน': 'Website', 'ออกบูทงานแสดงสินค้า': 'Trade-show booth',
}

TYPES = {'ซื้อซ้ำ': 'Repeat', 'สินค้าใหม่สำหรับลูกค้า': 'New for customer', 'ยอดนิยม (ลูกค้าใหม่)': 'Popular (new customer)'}

BASES = {
 'ชาเขียว': 'Green tea', 'อเมริกาโน่': 'Americano', 'น้ำเปล่า': 'Drinking water', 'ช็อคโกแลต': 'Chocolate', 'ลาเต้': 'Latte',
 'โกโก้': 'Cocoa', 'คาปูชิโน่': 'Cappuccino', 'ชาซีลอน': 'Ceylon tea', 'มาม่า': 'Mama instant noodles', 'เอสเปรสโซ่': 'Espresso',
 'อเมริกาโน่ส้ม': 'Orange Americano', 'นมสด': 'Fresh milk', 'นมสดคาราเมล': 'Caramel milk', 'ชามัทฉะ': 'Matcha tea',
 'แซนวิชโบราณ': 'Old-style sandwich', 'อเมริกาโน่ช่อมะพร้าว': 'Coconut-blossom Americano', 'มอคค่า': 'Mocha',
 'อเมริกาโน่น้ำผึ้ง': 'Honey Americano', 'ชามะนาว': 'Lemon tea', 'อเมริกาโน่น้ำผึ้งมะนาว': 'Honey-lemon Americano',
 'ชาพีช': 'Peach tea', 'ขนมซอง': 'Packaged snack', 'ขนมเลย์': "Lay's chips", 'ชาน้ำผึ้งมะนาว': 'Honey-lemon tea',
 'วุ้นบุก': 'Konjac jelly', 'โค้ก ออริจินอล': 'Coca-Cola Original', 'ป็อกกี้': 'Pocky', 'สตรอเบอรี่': 'Strawberry',
 'ยำยำ': 'Yum Yum noodles', 'น้ำแข็ง': 'Ice', 'คาราเมลมัคคีอาโต้': 'Caramel macchiato', 'บราวนี่': 'Brownie', 'เค้ก': 'Cake',
 'ไวไว': 'Wai Wai noodles', 'โจ๊ก': 'Congee', 'ขนมปังรวมรส': 'Assorted bread', 'สมูตตี้ สตรอเบอรี่': 'Strawberry smoothie',
 'เลม่อนผสมน้ำผึ้ง': 'Lemon with honey', 'ข้าวกล่อง': 'Boxed rice meal', 'แอลแคร์': 'L-Care', 'ชาแอปเปิ้ล': 'Apple tea',
 'ชากุหลาบนมสด': 'Rose milk tea', 'โกโก้มิ้นท์': 'Mint cocoa', 'นมชมพู(นมเย็น)': 'Pink milk (iced)', 'แตงโม': 'Watermelon',
 'เทสโต้': 'Testo snack', 'น้ำช่อดอกมะพร้าว': 'Coconut-blossom drink', 'ชาซีลอนดำ': 'Black Ceylon tea',
 'สิงห์เลม่อนโซดา': 'Singha lemon soda', 'ขนมจีบกุ้ง': 'Shrimp dumpling', 'ลิ้นจี่': 'Lychee', 'บานอฟฟี่': 'Banoffee',
 'พีช': 'Peach', 'ชาเขียวมะนาว': 'Green tea with lime', 'แดงมะนาว': 'Red syrup with lime', 'โอริโอ้': 'Oreo',
 'ชากุหลาบน้ำผึ้งมะนาว': 'Rose honey-lemon tea', 'น้ำอัดลม': 'Soft drink', 'โคอะล่ามาร์ช': "Koala's March",
 'ชาอัญชันมะนาว': 'Butterfly-pea lemon tea', 'แฟนต้า น้ำแดง': 'Fanta Red', 'ชาเขียวดำ': 'Green-black tea', 'โค้ก zero': 'Coke Zero',
 'ชาสตรอเบอรี่': 'Strawberry tea', 'มาการอง': 'Macaron', 'เค้กช็อคหน้านิ่ม': 'Soft-top chocolate cake', 'ขนมจีบไก่': 'Chicken dumpling',
 'กรือโป๊ะนรา': 'Narathiwat fish crackers', 'ขนม': 'Snack', 'กีวี่': 'Kiwi', 'สไปรท์': 'Sprite', 'ขนมปังกรอบ': 'Crispy bread',
 'ไข่กระทะ': 'Pan-fried eggs', 'บลูเบอรี่': 'Blueberry', 'เค้กบลูเบอร์รี่หน้านิ่ม': 'Soft-top blueberry cake', 'อาหารคลีน': 'Clean-eating meal',
 'คอนเน่': 'Conne corn snack', 'นมสดกล้วย': 'Banana milk', 'มินิเค้กช็อกโกแลต': 'Mini chocolate cake', 'สมูตตี้ บลูเบอรี่': 'Blueberry smoothie',
 'ซือดะรสต้มโคล้ง': 'Soda crackers (tom klong)', 'แอปเปิ้ล': 'Apple', 'ชเวปส์': 'Schweppes', 'โดนัท': 'Donut', 'สแน็กแจ็ค': 'Snack Jack',
 'มะม่วง': 'Mango', 'ลาเตั': 'Latte', 'ตะวัน': 'Tawan snack', 'สตรอเบอรี่นมสด': 'Strawberry milk', 'มะนาวโซดา': 'Lime soda',
 'ขนมซีมอน': 'Seemon snack', 'น้ำส้มคั้นสด': 'Fresh-squeezed orange juice', 'ขนมปัง': 'Bread', 'ชอตกาแฟ': 'Coffee shot', 'ฮานามิ': 'Hanami snack',
 'บราวนี่กรอบ': 'Crispy brownie', 'คอร์นเฟรก': 'Cornflakes', 'ขนมซีเรียลอาหารเช้า': 'Breakfast cereal snack', 'แฟนต้า น้ำส้ม': 'Fanta Orange',
 'ขนมปาตี้': 'Party snack', 'ชีสคัพ': 'Cheese cup', 'ปาปีก้า': 'Paprika chips', 'สมูตตี้ มะม่วง': 'Mango smoothie',
 'มินิเค้กสดสตรอเบอรี่': 'Mini fresh strawberry cake', 'ขนมเปี๊ยะ': 'Pia pastry', 'แฟนต้า น้ำเขียว': 'Fanta Green', 'โคอะลา มาร์ช': "Koala's March",
 'คอนเฟรก': 'Cornflakes', 'มัทฉะลาเต้': 'Matcha latte', 'น้ำส้มสด': 'Fresh orange juice', 'มินิเค้กมัทฉะครีมสด': 'Mini matcha cream cake',
 'เค้กชาไทยหน้านิ่ม': 'Soft-top Thai tea cake', 'คุกกี้ไข่เค็ม': 'Salted-egg cookie', 'เสาวรส': 'Passion fruit', 'สลัดโรง': 'Salad roll',
 'โปเต้': 'Potae chips', 'นิสชินคัพไก่เผ็ดเกาหลี': 'Nissin cup, Korean spicy chicken', 'แซนวิสสลัด': 'Salad sandwich', 'แดงโซดา': 'Red soda',
 'ขนมทวิสโก้': 'Twisko snack', 'กล้วนฉาบรวมรส': 'Assorted banana chips', 'แซนวิชไก่หยองโบโลน่า': 'Chicken-floss & bologna sandwich',
 'เค้กกาแฟหน้านิ่ม': 'Soft-top coffee cake', 'แซนวิชไก่หยอง': 'Chicken-floss sandwich', 'ตูมตาม': 'Tumtam snack',
 'เค้กเรดเวลเวทีราวนี่': 'Red velvet brownie cake', 'สมูตตี้ บลูฮาวาย': 'Blue Hawaii smoothie', 'มินิเค้กครีมสดบลูเบอร์รี่': 'Mini fresh-cream blueberry cake',
 'ข้าวต้ม': 'Rice porridge', 'สมูตตี้ กีวี่': 'Kiwi smoothie', 'บลูเบอรรี่': 'Blueberry', 'มินิเค้กชาไทย': 'Mini Thai tea cake',
 'คนอร์คัพโจ๊ก ไก่': 'Knorr congee cup, chicken', 'เค้กสตอเบอร์รี่หน้านิ่ม': 'Soft-top strawberry cake', 'คพุเค้ก': 'Cupcake',
 'ชาเขียวมัทฉะ': 'Matcha green tea', 'บันบันรสดั้งเดิม': 'Bun Bun original', 'ขนมทองม้วน': 'Thong muan egg rolls', 'แมกซ์เทสต์': 'Max Taste snack',
 'กล้วยอบน้ำผึ้ง': 'Honey baked banana', 'ท็อปปิ้งโอริโอ้': 'Oreo topping', 'บลูฮาวาย': 'Blue Hawaii', 'คนอร์คัพโจ็ก หมู': 'Knorr congee cup, pork',
 'เค้กจำปะดะ': 'Champadak cake', 'กล้วยเส้นคละรส': 'Assorted banana strips', 'ขนมของฝากคละรส': 'Assorted souvenir snacks',
 'ชาเขียวช่อมะพร้าว': 'Green tea with coconut blossom', 'น้ำส้ม': 'Orange juice', 'ปั้นขลิปรวมรส': 'Assorted Pan Khlip snacks', 'ก๊อบกอบ': 'Gobgob snack',
 'แซนวิชไก่': 'Chicken sandwich', 'เปี๊ยะไข่เค็ม': 'Salted-egg pia pastry', 'แก้วเยติ': 'Yeti tumbler',
 'อเมริกาโน่มะม่วงดองโซดาห์': 'Pickled-mango Americano soda', 'ขนมของฝากแบบถุง': 'Bagged souvenir snacks', 'นมสดโอริโอ้': 'Oreo milk',
 'โตเกียวกรอบนมสด': 'Crispy milk Tokyo snack', 'ยาคลู': 'Yakult', 'ขนมของฝาก': 'Souvenir snacks', 'องุ่นโซดา': 'Grape soda', 'ซันไบทส์': 'Sunbites',
 'แซนวิชไส้กรอก': 'Sausage sandwich', 'อาหารเจ': 'Vegetarian meal', 'แจ็กซ์': 'Jacks snack', 'ส้มโซดา': 'Orange soda', 'ปั้นขลิบ': 'Pan Khlip snack',
 'มาม่ารุสกี้': 'Mama Russky', 'มะนาว': 'Lime', 'ชาดำน้ำส้ม': 'Orange black tea', 'แซนวิชกล่อง': 'Boxed sandwich', 'บะหมี่จายารสต้มยำ': 'Jaya noodles, tom yum',
 'คุกกี้': 'Cookie', 'เบนโตะ': 'Bento snack', 'เค้กช้อกโกแลตลาวา': 'Chocolate lava cake', 'ปั้นขลิบห่อใส': 'Pan Khlip (clear wrap)', 'มะม่วงดอง': 'Pickled mango',
 'อกไก่รวมรส': 'Assorted chicken breast', 'ปั้นขลิบไส้ปลา กระปุก': 'Fish-filled Pan Khlip (jar)', 'องุ่นสมุทตี้': 'Grape smoothie', 'ปังเวอร์': 'Pangver snack',
 'มันฉาบ': 'Sweet-potato chips', 'สแตคส์(กระป๋อง)': 'Stax chips (can)', 'ขนมเปี๊ยะเล็ก': 'Small pia pastry', 'ทุเรียนทอด': 'Fried durian',
 'ปั้นขลิบถุง': 'Pan Khlip (bag)', 'เสาวรสโซดา': 'Passion fruit soda', 'อะโวคาโด้': 'Avocado', 'โจ๊กข้าวกล้อง': 'Brown rice congee',
 'มะม่วงหิมพานต์': 'Cashew nuts', 'เค้กชาเขียวหน้านิ่ม': 'Soft-top green tea cake', 'เผือกฉาบ': 'Taro chips', 'ขนมเค้กรวม': 'Assorted cakes',
 'ขนมปั้นขลิบ': 'Pan Khlip snack', 'มินิเค้กคัสตาร์ทครีมสด': 'Mini fresh custard-cream cake', 'มาม่าคัพออเรียลทัลฮอทสไปซี่': 'Mama cup, Oriental hot & spicy',
 'ไข่ปลาทอดกรอบ': 'Crispy fried fish roe', 'หมูแท่งกรอบ': 'Crispy pork sticks', 'ปั้นขลิบไส้ปลา แบบซอง': 'Fish-filled Pan Khlip (sachet)', 'ขนมไทย': 'Thai dessert',
 'ชาเขียวโอ๊ดมิลล์': 'Green tea with oat milk', 'ชเวปไม่มีน้ำตาล': 'Schweppes sugar-free', 'ข้าวเกรียบ': 'Rice crackers', 'เค้กกล้วยหอมเจ': 'Vegan banana cake',
 'เปี๊ยะกล่อง': 'Boxed pia pastry', 'น้ำอัญชัน': 'Butterfly-pea drink', 'บุกบราวชูการ์': 'Konjac brown sugar', 'มาม่าต้มยำกุ้งน้ำข้น': 'Mama creamy tom yum shrimp',
 'ป็อกกี้ช็อกโกแลต': 'Pocky chocolate', 'กรีบลำดวน': 'Lamduan cookie', 'ไวไวควิกต้มยำมันกุ้ง': 'Wai Wai Quick, tom yum shrimp-fat', 'เค้าคั่วกรอบ': 'Crispy roasted snack',
 'แซนวิช': 'Sandwich', 'คาราเมลมัคคิอาโต้': 'Caramel macchiato', 'อเมริกาโน่น้ำตาลโตนด': 'Palm-sugar Americano', 'ขนมกง': 'Khanom Kong', 'ผลไม้': 'Fruit',
 'ป็อปคอน': 'Popcorn', 'ยำยำต้มยำกุ้งน้ำข้น': 'Yum Yum creamy tom yum shrimp', 'บันบันรสปาปริก้า': 'Bun Bun paprika', 'บราวนี่เจ': 'Vegan brownie',
 'ขรัวหน้าตัง': 'Crispy rice cracker', 'ขนมเทสโตพริกจักรพรรดิ์': 'Testo imperial chili snack', 'เค้กส้ม': 'Orange cake',
 'โคอาล่ามาร์ชรสสตรอเบอรี่': "Koala's March, strawberry", 'ปั้นขลิบไส้ไก่หยอง แบบซอง': 'Chicken-floss Pan Khlip (sachet)', 'วิปปิ้งครีม': 'Whipped cream',
 'สลัดโรล': 'Salad roll', 'สตอเบอร์รี่อัลมอนด์': 'Strawberry almond', 'มาม่าผัดทะเล': 'Mama seafood stir-fry noodles', 'เค้กโกไข่': 'Cocoa egg cake',
 'เค้กส้มหน้านิ่ม': 'Soft-top orange cake', 'น้ำอัญชันมะนาวสด': 'Fresh butterfly-pea lime drink', 'ไวไวควิกกุ้งนึ่งมะนาว': 'Wai Wai Quick, steamed shrimp lime',
 'น้ำเชื่อมต่างๆ': 'Assorted syrups', 'ปาตี้รสคาราเมล': 'Party snack, caramel', 'ลินจี่': 'Lychee', 'คอนเน่ข้าวโพด': 'Conne corn snack',
 'น้ำเคยแห้ง': 'Dried shrimp-paste dip', 'ไวไวควิกรสต้มยำพริกเผา': 'Wai Wai Quick, tom yum chili paste', 'แจ็กซ์น้ำจิ้มลูกชิ้น': 'Jacks, meatball-dip flavor',
 'ปั้นขริป': 'Pan Khlip snack', 'เทสโต้บาร์บิคิว': 'Testo BBQ', 'ไวไวควิกถ้วยต้มโคล้ง': 'Wai Wai Quick cup, tom klong', 'องุ่น': 'Grape',
 'บันบันรสปาปีก้า': 'Bun Bun paprika', 'ก๊อบกอบรสต้มยำกุ้ง': 'Gobgob, tom yum shrimp', 'ลาเต้อัลมอนด์': 'Almond latte', 'ยำยำต้มยำกุ้ง': 'Yum Yum tom yum shrimp',
 'มะม่วงดองเกลือ': 'Salted pickled mango', 'สแน็กแจ็คถั่วลันเตา': 'Snack Jack, green peas', 'เลย์น้ำพริกกุ้งทรงเครื่อง': "Lay's shrimp chili paste",
 'เลย์สแตนส์รสชาวทวิน': "Lay's Stax", 'เลย์บาบีคิว': "Lay's BBQ", 'ปั้นขลิปใส้ไก่': 'Chicken-filled Pan Khlip', 'วุ้นหมี': 'Gummy bears',
 'มาม่ารสหมูสับ': 'Mama minced pork', 'มิกเบอร์รี่ สมูตตี้เจ': 'Mixed-berry smoothie (vegan)', 'ช็อคโกแลตบุก': 'Chocolate with konjac',
 'สแน็คแจคกุ้งพริกเกลือ': 'Snack Jack, salt-chili shrimp', 'ตะวันลูกชิ้นปิ้ง': 'Tawan grilled-meatball snack', 'ขนมข้าวเกรียบกุ้ง': 'Prawn crackers',
 'บิสกิตชีสเค้ก': 'Cheesecake biscuit', 'สแน็คแจ็รสดั้งเดิม': 'Snack Jack original', 'สแน็คแจ็ครสดั้งเดิม': 'Snack Jack original',
 'ป็อกกี้แฮปปี้เนส': 'Pocky Happiness', 'ปาปิก้ามันฝรั่ง': 'Papika potato chips', 'มาม่าเผ็ดเกาหลี': 'Mama Korean spicy',
 'โปเต้มันฝรั่งทอด': 'Potae fried potato chips', 'ยำยำสุกกี้ทะเล': 'Yum Yum seafood sukiyaki', 'ร่ม': 'Umbrella', 'น้ำมะม่วงดอง': 'Pickled-mango drink',
 'น้ำกระเจี๊ยบ': 'Roselle juice', 'เค้กกล้วยหอม': 'Banana cake', 'นมสดปั่น บุก': 'Blended fresh milk with konjac', 'นมสด เย็น บุก': 'Iced fresh milk with konjac',
 'เซตแซนวิชสลัด': 'Sandwich & salad set', 'เทสโต้รสชีทซี่บาบีคิว': 'Testo cheesy BBQ', 'เค้กมะพร้าวอ่อนเจ': 'Vegan young-coconut cake', 'ชุดแก้ว': 'Cup set',
 'ชาเขียวปั่น บุก': 'Blended green tea with konjac', 'นมสดคาราเมลบุก': 'Caramel milk with konjac', 'ชาซีลอนบุก': 'Ceylon tea with konjac',
 'ช็อคโกแลต เย็นบุก': 'Iced chocolate with konjac', 'เลย์เครื่องยำ': "Lay's Yum spice", 'เลย์เรียบรสต้มยำหม้อไฟ': "Lay's Classic, tom yum hot-pot",
 'เห็ดทอด': 'Fried mushrooms', 'ขนมไทยหวาน': 'Sweet Thai dessert', 'ชามัทฉะ เย็นบุก': 'Iced matcha tea with konjac', 'ข้าวเหนียวดำแก้ว': 'Black sticky rice cup',
 'คนอร์ข้าวต้มกุ้งกระเทียม': 'Knorr garlic-shrimp porridge', 'ขนมปังเนยสด': 'Butter bread', 'แซนวิชไก่หยองโลโลน่า': 'Chicken-floss & bologna sandwich',
 'แซนวิส': 'Sandwich', 'ไวไสควิกหมูสับต้มยำ': 'Wai Wai Quick, tom yum minced pork', 'ไวไวควิกหมูสับต้มยำ': 'Wai Wai Quick, tom yum minced pork',
 }

_FORM_RE = re.compile(r'^(.*?)\s*\((ร้อน|เย็น|ปั่น)\)$')


def name_en(s):
    if not isinstance(s, str): return s
    if s in BASES: return BASES[s]
    m = _FORM_RE.match(s)
    if m:
        b = m.group(1).strip()
        return f"{BASES[b]} ({FORMS[m.group(2)]})" if b in BASES else s
    return s


def category_en(s): return CATEGORIES.get(s, s) if isinstance(s, str) else s
def channel_en(s): return CHANNELS.get(s, s) if isinstance(s, str) else s
def type_en(s): return TYPES.get(s, s)
