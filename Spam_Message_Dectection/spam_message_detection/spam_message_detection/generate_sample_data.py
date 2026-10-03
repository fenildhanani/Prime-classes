"""Creates a SYNTHETIC SMS dataset (3000 rows) so you can run the project immediately.
For real results, download the real 'SMS Spam Collection' (see README) and use that instead."""
import random
import pandas as pd

random.seed(42)
names = ["Rahul", "Priya", "Amit", "Sneha", "John", "Maria", "Ali", "Neha", "Tom", "Sara"]
places = ["office", "home", "station", "mall", "college", "airport", "cafe", "gym"]
times = ["5pm", "tomorrow", "tonight", "9am", "this weekend", "in 10 mins", "after lunch"]
brands = ["Amazon", "PayPal", "HDFC", "SBI", "Netflix", "Apple", "Paytm", "FedEx"]
prizes = ["iPhone 15", "$1000 cash", "a free holiday", "Rs 50000", "a gift voucher", "a new laptop"]
codes = ["WIN", "CLAIM", "STOP", "YES", "OFFER"]

ham_t = [
    "Hey {n}, are you coming to the {p} {t}?", "Ok see you {t} at the {p}",
    "Can you call me when you reach {p}? Need to ask something",
    "Happy birthday {n}! Have a great day", "I will be late, stuck in traffic near {p}",
    "Did you finish the assignment? Please send it by {t}", "Dinner at my place {t}? Mom is cooking",
    "Thanks for the help yesterday {n}, really appreciate it", "Meeting moved to {t}, please confirm",
    "lol that was so funny, tell me more when we meet", "Don't forget to buy milk on the way home",
    "Call me when you are free, nothing urgent", "Can I borrow your notes? Will return {t}",
    "Your order is out for delivery, it will arrive {t}",              # legit-looking, tricky ham
    "Your OTP for login is 483920. Do not share it with anyone",      # legit-looking, tricky ham
    "Free for a quick call {t}? Want to discuss the project",         # contains 'free'
    "Congrats {n} on the new job! Let's celebrate {t}",               # contains 'congrats'
    "Reminder: your appointment is {t}. Reply YES to confirm",
]
spam_t = [
    "WINNER!! You have been selected to receive {z}. Call 0{d} now to claim",
    "Congratulations! You won {z}. Click http://claim-{b}.com/win to collect your prize",
    "URGENT: Your {b} account is suspended. Verify now at www.{b}-secure.net/login",
    "FREE entry in our weekly draw to win {z}! Text {c} to {s} now. T&Cs apply",
    "You have been specially selected for {z}. Reply {c} to {s} to claim before midnight",
    "Dear customer, your {b} KYC is expired. Update immediately: http://{b}-kyc.info or account will be blocked",
    "Get a loan of Rs {a} approved instantly. No documents needed! Call 0{d}",
    "Hot singles in your area want to chat. Text {c} to {s}. 18+ only",
    "Claim your FREE {z} now!!! Limited time offer, click www.{b}-offers.com",
    "Your mobile number has won £{a} in the lottery. Contact agent on 0{d} to receive payment",
    "Final notice: pay outstanding bill of ${a} today to avoid legal action. Visit http://pay-{b}.biz",
    "Earn $500 daily working from home. No experience needed. Join now www.easy-cash.com",
    "Exclusive offer from {b}: 90% OFF today only. Shop now http://{b}-deals.com",
    "Hi, I saw your profile. Invest in crypto and double your money in 7 days. Message me",   # subtle spam
    "Your parcel could not be delivered. Pay small fee to reschedule: http://{b}-track.info",
]

def fill(t):
    return t.format(n=random.choice(names), p=random.choice(places), t=random.choice(times),
                    b=random.choice(brands).lower(), z=random.choice(prizes),
                    c=random.choice(codes), s=random.randint(50000, 89999),
                    d=random.randint(7000000000, 9999999999), a=random.randint(500, 9000), r='{r}')

ham_pre = ["", "", "Hi, ", "Hey! ", "Hello {n}, ", "Btw ", "Ok so ", "Hi {n}! ", "Quick one: ", "Morning. "]
ham_post = ["", "", " Let me know.", " Thanks!", " See you.", " Talk later.", " Cheers {n}", " Bye", " Ping me.", " ok?"]
spam_pre = ["", "", "ALERT: ", "Dear user, ", "Hello, ", "Notice: ", "Hi {n}, ", "Important! ", "FREE MSG: ", "Attention: "]
spam_post = ["", "", " Reply STOP to opt out.", " T&Cs apply.", " Hurry, offer ends soon!", " Act now!", " Ref {r}", " Valid 24hrs only.", " Do not ignore!", " Msg ID {r}"]

def vary(t, pre, post):
    return fill(random.choice(pre)) + t + fill(random.choice(post)).replace("{r}", str(random.randint(1000, 99999)))

rows = []
for _ in range(2610):                      # ~87% ham
    m = vary(fill(random.choice(ham_t)), ham_pre, ham_post)
    if random.random() < .25: m = m.lower()
    if random.random() < .15: m += random.choice([" :)", " ok?", " thanks", " pls"])
    rows.append(("ham", m))
for _ in range(390):                       # ~13% spam
    m = vary(fill(random.choice(spam_t)), spam_pre, spam_post)
    if random.random() < .30: m = m.lower()
    rows.append(("spam", m))

# add ~1.5% label noise (real datasets are never perfectly clean)
rows = [(("spam" if l == "ham" else "ham") if random.random() < .015 else l, m) for l, m in rows]
random.shuffle(rows)
pd.DataFrame(rows, columns=["label", "text"]).to_csv("data/sample_spam.csv", index=False)
print("Saved data/sample_spam.csv with", len(rows), "rows")
