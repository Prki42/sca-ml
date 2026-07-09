# %% [markdown]
# ### Uvod: Side-channel Analysis
# Side-channel analysis (skraćeno SCA) je klasa kriptoanalitičkih metoda napada koja se bavi analizom fizičkih merenja nekog kriptografskog uređadja, najčešće vreme izvršavanja, potrošnje struje ili elektromagnetnog zračenja, da bi se dobila informacija o parametrima enkripcije, najpre o ključu koji se koristi za enkripciju. SCA napadi se mogu svrstati u dve grupe: profilišuće (profiling) i neprofilišuće (non-profiling). 
#
# Profilišući napadi se sastoje iz dvaju koraka: prvo, napadač ima pristup kopiji uređaja koji vrši enkripciju, i može na njemu da sprovodi gorenavedena fizička merenja (engl. leakage, ,,curenje"). U drugom koraku, napadač koristi dobijeno znanje na "ciljnom" uređaju, u pokušaju saznanja ključa koji se koristi za enkripciju na tom uređaju. Dakle, u pitanju je **problem klasifikacije**.
#
# Neprofilišući napadi su slabiji, jer napadač nema kopiju ciljnog uređaja, već samo direktan pristup samom tom uređaju. On može da vrši određene statističke analize da uspostavi neku korelaciju između curenja (merenja) i ciljne promenljive (ključa). Mi ćemo se u ovom projektu baviti isključivo profilišućim napadima, i stoga ćemo imati i skupove $\mathcal{D}_{profiling}$ i $\mathcal{D}_{attack}$, koji odgovaraju trening i test skupovima za naše modele. O njima će biti detaljnije rečeno docnije.
#
# TODO: OVDE NEŠTO O ASCADU I AESU

# %% [markdown]
# ### Oznake i statističko definisanje problema
# Sa $\vec{L}$ označavamo vektor merenja, prikupljenih u diskretnim trenutcima tokom određenog vremena. Suštinski, to je vremenska serija. Sa $K$ označavamo ključ (odnosno deo ključa) koji se koristi za enkripciju, a sa $P$ označavamo skup plaintext-a, odnosno njegov podskup određene dužine (kod nas jedan bajt). U profilišućim napadima, napadač, koristeći svoju kopiju uređaja, razvija model kojim za svaki mogući ključ $k \in K$ želi da aproksimira uslovnu verovatnoću $\vec{g}[k]:(\vec{l},p)\mapsto P\{K=k \mid (\vec{L},P)=(\vec{l},p)\}$ nekom ocenom $\hat{\vec{g}}[k]$.
#
# Ključna ideja je da model ne posmatra ključ direktno - on kao ulaz dobija isključivo vektor merenja $\vec{l}$. Potrošnja struje uređaja zavisi od međuvrednosti koje se javljaju tokom izvršavanja algoritma, a ne od samog ključa u izolaciji. U prvoj rundi AES-a, uređaj za svaki bajt ključa računa $Sbox(p \oplus k)$, gde je $p$ poznati plaintext a $k$ nepoznati ključ. Ova operacija zavisi od oba parametra i uzrokuje merljive obrasce u potrošnji struje. Stoga model treniramo da na osnovu merenja predvidi vrednost ove posredne funkcije (koja uzima jednu od 256 mogućih vrednosti), a potom pri napadu za svaki poznati plaintext isprobavamo svih 256 kandidata za ključ i proveravamo koji od njih daje predviđanja najkonzistentnija sa modelom. To jest, tražimo $\vec{g}[z]:(\vec{l}, p) \mapsto P\{Z = f(p, K) \mid (\vec{L},P)=(\vec{l},p)\}$ gde je $Z$ međuvrednost koja se dobija funkcijom $f$ za dat plaintext $P$ i ključ $K$.
#
# Sa bazom koju ćemo koristiti u ovom projektu, predviđaćemo vrednost međuvrednosti $Sbox(p\oplus k)$, odnosno razvijaćemo ocenu za
# $$
#     \vec{g}[Sbox(p\oplus k)]:(\vec{l},p)\mapsto P\{Sbox(p\oplus k) = Sbox(p\oplus K) \mid (\vec{L},P)=(\vec{l},p)\}
# $$
# kada $Sbox(p\oplus K)$ prolazi vrednosti $\{0,1, \dots ,255\}$.

# %% [markdown]
# ### Napad
# Tokom faze napada (nakon treniranja gorenavedenog modela), napadač želi da sazna nepoznati ključ $k^∗$ koji uredaj koji on napada koristi za šifrovanje, a na raspolaganju ima poznate plaintextove $p$ i, naravno, njegov istreniran model. Napadač želi da utvrdi koja od njegovih funkcija k je najverovatnija, s obzirom na merenja koja izvršava nad uređajem koji napada. Ta merenja koja koristimo u ovoj fazi označavamo sa $\mathcal{D}_{attack}$, pri čemu $N_a = |\mathcal{D}_{attack}|$.
#
# Tu se prirodno javlja kao ocena baš ona dobijena metodom maksimalne verodostojnosti, dakle maksimiziranjem funkcije:
# $$
# \begin{align}
#     \vec{d_{N_a}}[k] &= \prod_{i=1}^{N_a}P\{K=k \mid (\vec{L},P)=(\vec{l_i},p_i)\} \\
#     &=\prod_{i=1}^{N_a}P\{Sbox(K\oplus p_i)=Sbox(k\oplus p_i) \mid (\vec{L},P)=(\vec{l_i},p_i)\}
# \end{align}
# $$
# koju ocenjujemo sa
# $$\prod_{i=1}^{N_a}\hat{\vec{g}}(\vec{l_i},p_i)[k]$$
# Drugim rečima, naše predviđanje za tačan ključ pri napadu će biti
# $$\hat{k}=\arg\max_{k \in \{0,...,255\}} \prod_{i=1}^{N_a}\hat{\vec{g}}(\vec{l_i},p_i)[k]$$
# Princip maksimalne verodostojnosti ćemo koristiti i u rang funkciji za evaluaciju modela, o čemu će biti reči kasnije.

# %% [markdown]
# ### Evaluacija
# Prirodno okruženje za evaluaciju našeg modela je upravo okruženje u kojem napadamo novi uređaj, odnosno test skup je baza u kojoj je kljuć takođe konstantan. Ovde se ugleda važnost treniranja modela na samo jednom ključu, iako to na prvi pogled deluje neintuitivno, jer generalno trening skupovi treba da budu što raznovrsniji. Međutim, u našoj primeni nije tako - model koji treniramo suštinski treba da upamti obrasce koje vezuju neku međuvrednost i merenja, i u tom smislu "šum" od različitih ključeva ne bi nikako odgovarao.
#
# Najčešća mera performansi ovakvog modela je takozvana "rang funkcija". Do nje dolazimo na sasvim prirodan način, samo što ne posmatramo svih $N_a$ opservacija iz test skupa, već nekih $i$. Zbog monotonosti logaritamske funkcije, važi:
# $$
# \begin{align}
#     \arg\max_{k \in \{0,...,255\}} \prod_{t=1}^{i}\hat{\vec{g}}(\vec{l_t},p_t)[k] &= \arg\max_{k \in \{0,...,255\}} \ln \prod_{t=1}^{i}\hat{\vec{g}}(\vec{l_t},p_t)[k]\\
#     &= \arg\max_{k \in \{0,...,255\}} \sum_{t=1}^{i} \ln \hat{\vec{g}}(\vec{l_t},p_t)[k]\\
#     &= \arg\max_{k \in \{0,...,255\}} \hat{S_i}[k]
# \end{align}
# $$
# gde je $S$ tako zvana (nagomilana, kumulativna) _skor_ funkcija. Razlog zbog čega koristimo logaritam je dvostruki: kako je logaritam strogo rastuć, on neće promeniti ocenu maksimalne verodostojnosti (a ni poredak između predviđenih verovatnoća), a sa numeričke strane, množenje velikog broja predviđenih verovatnoća će dati veoma male brojeve, pa može doći do takozvanog "underflow-a".
#
# Najzad, pređimo na rang funkciju. Ako sa $k^*$ definišemo ključ korišćen u $\mathcal{D}_{attack}$, tada definišemo rang kao: 
# $$ rank_i(k^*) = |\{k\in K \mid \hat{S_i}(k) > \hat{S_i}(k^*)\}| $$
# Drugim rečima, to je broj ključeva čija je funkcija verodostojnosti (zajednička, na prvih $i$ opservacija) veća od funkcije verodostojnosti za pravi ključ. Jasno, mi želimo da rang pravog ključa $k^*$ bude što manji, i to što brže - dakle, da model na osnovu što manjeg broja merenja dođe što bliže pravom ključu.
#
# TODO: ocekivana rang funkcija
