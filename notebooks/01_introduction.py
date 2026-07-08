# %% [markdown]
# ### Uvod: Side-channel Analysis
# Side-channel analysis (skraćeno SCA) je klasa kriptoanalitičkih metoda napada koja se bavi analizom fizičkih merenja nekog kriptografskog uređadja, najčešće vreme izvršavanja, potrošnje struje ili elektromagnetnog zračenja, da bi se dobila informacija o parametrima enkripcije, najpre o ključu koji se koristi za enkripciju. SCA napadi se mogu svrstati u dve grupe: profilišuće (profiling) i neprofilišuće (non-profiling). 
#
# Profilišući napadi se sastoje iz dvaju koraka: prvo, napadač ima pristup kopiji uređaja koji vrši enkripciju, i može na njemu da sprovodi gorenavedena fizička merenja (engl. leakage, ,,curenje"). U drugom koraku, napadač koristi dobijeno znanje na "ciljnom" uređaju, u pokušaju saznanja ključa koji se koristi za enkripciju na tom uređaju. Dakle, u pitanju je **problem klasifikacije**.
#
# Neprofilišući napadi su slabiji, jer napadač nema kopiju ciljnog uređaja, već samo direktan pristup samom tom uređaju. On može da vrši određene statističke analize da uspostavi neku korelaciju između curenja (merenja) i ciljne promenljive (ključa). Mi ćemo se u ovom projektu baviti isključivo profilišućim napadima, i stoga ćemo imati i skupove $\mathcal{D}_{profiling}$ i $\mathcal{D}_{attack}$, koji odgovaraju trening i test skupovima za naše modele. O njima će biti detaljnije rečeno docnije.
#
# TODO: OVDE NEŠTO O NASOJ BAZI I ASCADU

# %% [markdown]
# ### Oznake i statističko definisanje problema
# Sa $\vec{L}$ označavamo vektor merenja, prikupljenih u diskretnim trenutcima tokom određenog vremena. Suštinski, to je vremenska serija. Sa $K$ označavamo ključ (odnosno deo ključa) koji se koristi za enkripciju, a sa $P$ označavamo skup plaintext-a, odnosno njegov podskup određene dužine (kod nas jedan bajt). U profilišućim napadima, napadač, koristeći svoju kopiju uređaja, razvija model kojim za svaki mogući ključ $k \in K$ želi da aproksimira uslovnu verovatnoću $\vec{g}[k]:(\vec{l},p)\mapsto P\{K=k|(\vec{L},P)=(\vec{l},p)\}$ nekom ocenom $\hat{\vec{g}[k]}$. Sa bazom koju ćemo koristiti u ovom projektu, nećemo direktno imati kao uslov $(P,K)=(p,k)$, već ćemo predviđati vrednost posredne funkcije $Sbox(p\oplus k)$, odnosno razvijaćemo ocenu za $$\vec{g}[Sbox(p\oplus k)]:(\vec{l},p)\mapsto P\{Sbox(p\oplus k) = Sbox(p\oplus K)|(\vec{L},P)=(\vec{l},p)\}$$, kada $Sbox(p\oplus K)$ prolazi vrednosti $\{0,1,...,255\}$.

# %% [markdown]
# ### Napad
# Tokom faze napada (dakle, nakon treniranja gorenavedenog
# modela), napadač želi da sazna nepoznati ključ $k^∗$ koji uredaj koji on napada
# koristi za šifrovanje, a na raspolaganju ima poznate plaintextove $p$ i, naravno, njegov model. Napadač dakle želi da utvrdi koja od njegovih funkcija k
# je najverovatnija, s obzirom na merenja koja izvršava nad uređajem koji napada (koja dakle čine skup $\mathcal{D}_{attack}$). Tu se prirodno javlja kao ocena baš ona dobijena metodom maksimalne
# verodostojnosti, dakle maksimiziranjem funkcije:
# $$\begin{align}
#     \vec{d_{N_a}}[k] &= \prod_{i=1}^{N_a}P\{(K=k|(\vec{L},P)=(\vec{l_i},p_i)\} \\
#     &=\prod_{i=1}^{N_a}P\{Sbox(K\oplus p_i)=Sbox(k\oplus p_i)|(\vec{L},P)=(\vec{l_i},p_i)\}
#     \end{align}$$
# koju naravno ocenjujemo sa
# $$\prod_{i=1}^{N_1}\hat{\vec{g}}(\vec{l_i},p_i)[k]$$
# Drugim rečima, naše predviđanje za tačan ključ pri napadu će biti
# $$\hat{k}=\arg\max_{k \in \{0,...,255\}} \prod_{i=1}^{N_1}\hat{\vec{g}}(\vec{l_i},p_i)[k]$$
# pri čemu $N_a = |\mathcal{D}_{attack}|$. Princip maksimalne verodostojnosti ćemo koristiti i u rang funkciji za evaluaciju modela, o čemu će biti reči kasnije.

# %% [markdown]
# ### Evaluacija
# Prirodno okruženje za evaluaciju našeg modela je upravo okruženje u kojem napadamo novi uređaj, odnosno test skup je baza u kojoj je kljuć nepoznat, ali takođe konstantan. Ovde se ugleda važnost treniranja modela na samo jednom ključu, iako to na prvi pogled deluje neintuitivno, jer generalno trening skupovi treba da budu što raznovrsniji. Međutim, u našoj primeni nije tako - model koji treniramo suštinski treba da upamti obrasce koje vezuju **plaintext** i merenja, i u tom smislu "šum" od različitih ključeva ne bi nikako odgovarao.
#
# Najčešća mera performansi ovakvog modela je tako zvana "rang funkcija". Do nje dolazimo na sasvim prirodan način, samo što ne posmatramo svih $N_a$ opservacija iz test skupa, već nekih $i$. Zbog monotonosti logaritamske funkcije, važi:
# $$ \begin{align}
# \arg\max_{k \in \{0,...,255\}} \prod_{t=1}^{i}\hat{\vec{g}}(\vec{l_t},p_t)[k] &= \arg\max_{k \in \{0,...,255\}} \ln \prod_{t=1}^{i}\hat{\vec{g}}(\vec{l_t},p_t)[k]\\ &= \arg\max_{k \in \{0,...,255\}} \sum_{t=1}^{i} \ln \hat{\vec{g}}(\vec{l_t},p_t)[k]\\ &= \arg\max_{k \in \{0,...,255\}} \hat{S_i}[k]
# \end{align} $$
# gde je S tako zvana (nagomilana, kumulativna)   _skor_ funkcija. Razlog zbog čega koristimo logaritam je dvostruki: kako je logaritam strogo rastuć, on neće promeniti ocenu maksimalne verodostojnosti (a ni poredak između predviđenih verovatnoća), a s numeričke strane, množenje velikog broja predviđenih verovatnoća će dati veoma male brojeve, pa može doći do tako zvanog "underflow-a".
#
# Najzad, pređimo na rang funkciju. Ako sa $k^*$ definišemo ključ korišćen u $\mathcal{D}_{attack}$, tada definišemo rang kao: 
# $$ rank_i(k^*) = |\{k\in K|\hat{S_i}(k) > \hat{S_i}(k^*)\}| $$
# Drugim rečima, to je broj ključeva čija je funkcija verodostojnosi (zajednička, na prvih $i$ opservacija) veća od funkcije verodostojnosti za pravi ključ. Jasno, mi želimo da rang pravog ključa $k^*$ bude što manji, i to što brže - dakle, da model na osnovu što manjeg broja merenja dođe do pravog ključa. 
#
# TODO: ocekivana rang funkcija
