"""Builds data/seed/players_2026_27.csv from prices published in public articles.

This is the *demo* player pool used when the official UCL Fantasy feed is not reachable.
Every row carries the article it came from (see data/seed/price_sources.csv).
When `ufe sync` can read the official feed, the feed replaces this file entirely.

Notes on conflicts between sources are recorded in the `note` column.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLUB = {
    "AEK Athens": "AEK", "AEK": "AEK", "Aston Villa": "AVL", "Atletico Madrid": "ATM", "Atleti": "ATM",
    "Barcelona": "BAR", "Bayern": "BAY", "Bayern Munich": "BAY", "Dortmund": "BVB", "Bodo/Glimt": "BOD",
    "Club Brugge": "CLB", "Como": "COM", "Feyenoord": "FEY", "Galatasaray": "GAL", "Inter": "INT",
    "LASK": "LAS", "Leipzig": "RBL", "Lens": "RCL", "Lille": "LIL", "Liverpool": "LIV", "Man City": "MCI",
    "Man United": "MUN", "Napoli": "NAP", "PSG": "PSG", "Porto": "POR", "PSV": "PSV", "Real Betis": "BET",
    "Roma": "ROM", "Sabah": "SAB", "Slovan Bratislava": "SLO", "Shakhtar": "SHA", "Slavia Praha": "SLA",
    "Sporting": "SCP", "Stuttgart": "VFB", "Viking": "VIK", "Villarreal": "VIL", "Arsenal": "ARS",
    "Real Madrid": "RMA", "Fenerbahce": "FEN",
}

# --- Fantasy Football Scout, "Assessing 200 budget defenders" (1 Sep 2026) -------------------
DEF_400 = """AEK Athens:Martin Georgiev
Aston Villa:Lamare Bogarde
Atletico Madrid:Marc Pubill
Barcelona:Gerard Martín;Héctor Fort
Bayern:Nathaniel Brown;Sacha Boey;Hiroki Ito
Dortmund:Filippo Mane;Luca Reggiani;Joane Gadou
Bodo/Glimt:Haitam Aleesami
Club Brugge:Kyriani Sabbe;Hugo Siquet;Jorne Spileers;Lee Han-beom
Como:Yan Couto;Ignace Van der Brempt;Alex Valle;Ivan Smolčić;Jacobo Ramón;Marc Oliver Kempf;Kaiki Bruno;Edoardo Goldaniga;Andrés Cuenca
Feyenoord:Gijs Smal;Bart Nieuwkoop;Jordan Lotomba;Mats Deijl;Thomas Beelen
Galatasaray:Wilfried Singo;Eren Elmalı;Kaan Ayhan;Metehan Baltacı
Inter:Carlos Augusto
LASK:Manoel Verhaeghe
Leipzig:Lukas Klostermann;Benjamin Henrichs;Max Finkgräfe;Maxime Estève;El Chadaille Bitshiabu
Lens:Nidal Celik;Kyllian Antonio;Saud Abdulhamid
Lille:Calvin Verdonk;Ousmane Touré;Tiago Santos;Tanguy Nianzou
Liverpool:Joe Gomez;Kostas Tsimikas
Man City:Rico Lewis;Vitor Reis
Man United:Leny Yoro;Ayden Heaven
Napoli:Alessandro Buongiorno;Mathías Olivera;Sam Beukema;Pasquale Mazzocchi
PSG:Illia Zabarnyi;Lucas Beraldo
Porto:Zaidu Sanusi;Nehuén Pérez;Martim Fernandes;Francisco Moura
PSV:Mauro Júnior;Yarek Gasiorowski;Armando Obispo;Joel Berg;Adamo Nagalo;Fabian Merien
Real Betis:Valentín Gómez;Natan
Roma:Jan Ziółkowski;Devyne Rensch;Daniele Ghilardi
Sabah:Aden McCarthy;Tellur Mutallimov
Slovan Bratislava:Samuel Kozlovský;Kevin Wimmer
Shakhtar:Pedro Henrique;Vinícius Tobias;Marlon Santos;Oleksandr Karavaiev;Giorgi Gocholeishvili;Alaa Ghram;Irakli Azarovi;Diego Arroyo
Slavia Praha:Youssoupha Mbodji;Štěpán Chaloupek;Tomáš Vlček;Daiki Hashioka;Ange N'Guessan;Neil Glossoa;Samuel Kelechi;Dominik Javorček;David Jurásek
Sporting:Iván Fresneda;Sergi Altimira
Stuttgart:Ameen Al-Dakhil;Lorenz Assignon;Ramon Hendriks;Maximilian Herwerth;Luca Jaquez;Dominik Nothnagel;Leonidas Stergiou;Josha Vagnoman;Dan-Axel Zagadou
Viking:Anders Bærtelsen;Sondre Bjørshol;Martin Røseth;Jesper Daland;Viljar Vevatne;Kristoffer Haugen
Villarreal:Santiago Mouriño;Sergi Cardona;Alex Freeman;Willy Kambwala;Carlos Romero"""

DEF_450 = """AEK Athens:Filipe Relvas;Harold Moukoudi;Stavros Pilios;Lazaros Rota
Arsenal:Piero Hincapié;Myles Lewis-Skelly;Riccardo Calafiori;Ezri Konsa
Aston Villa:Matteo Ruggeri;Matty Cash;Victor Lindelöf;Ian Maatsen;Tyrone Mings;Pau Torres
Atleti:David Hancko;Robin Le Normand
Barcelona:Eric García;Alejandro Balde;Andreas Christensen
Bayern:Josip Stanišić;Kim Min-jae
Dortmund:Ramy Bensebaini;Waldemar Anton;Daniel Svensson;Emre Can
Bodo/Glimt:Fredrik André Bjørkan;Villads Nielsen
Club Brugge:Brandon Mechele;Joaquin Seys;Joel Ordóñez
Feyenoord:Anel Ahmedhodžić;Givairo Read;Jerry Juste;Tsuyoshi Watanabe
Galatasaray:Davinson Sánchez;Abdülkerim Bardakcı;Ismail Jakobs
Inter:Yann Bisseck;Benjamin Pavard
LASK:Andrés Andrade;George Bello
Leipzig:Lutsharel Geertruida;Castello Lukeba;Willi Orbán;David Raum
Lens:Rúben Aguilar;Samson Baidoo;Jhoanner Chávez;Ismaëlo Ganiou;Jonathan Gradit;Matthieu Udol
Lille:Alexsandro;Nathan Ngoy;Romain Perraud;Loïc Sradanovic
Liverpool:Conor Bradley;Ronald Araújo;Giovanni Leoni
Man City:Abdukodir Khusanov
Man United:Matthijs de Ligt;Diogo Dalot;Lisandro Martínez;Luke Shaw;Noussair Mazraoui
Napoli:Leonardo Spinazzola;Amir Rrahmani;Rafa Marín;Luca Marianucci
PSG:Lucas Hernández;Lucas Digne
Porto:Alberto Costa;Jan Bednarek;Jakub Kiwior;Dominik Prpić
PSV:Ryan Flamingo;Kiliann Sildillia
Real Betis:Fran García;Marc Bartra;Héctor Bellerín;Diego Llorente;Junior Firpo
Real Madrid:Dean Huijsen;Raúl Asencio;Éder Militão;Ferland Mendy
Roma:Angeliño;Mario Hermoso;Gianluca Mancini;Evan Ndicka;Wesley França
Sabah:Steve Solvet;Akim Zedadka
Slovan Bratislava:Kenan Bajrić;César Blackman;Svetozar Marković;Sandro Cruz
Shakhtar:Valeriy Bondar;Mykola Matviyenko
Slavia Praha:David Zima;Tomáš Holeš;Igoh Ogbu
Sporting:Gonçalo Inácio;Eduardo Quaresma;Zeno Debast;Georgios Vagiannidis
Stuttgart:Jeff Chabot;Alexander Groß;Maximilian Mittelstädt;Christopher Olivier;Yanik Spalt
Viking:Henrik Falchener;Herman Haugen;Henrik Heggheim;Gianni Stensness
Villarreal:Renato Veiga;Juan Foyth;Pau Navarro;Logan Costa"""

# name, club, pos, price, source, note
OTHERS = [
    # goalkeepers
    ("David Raya", "Arsenal", "GK", 6.0, "ffs_premium", ""),
    ("Thibaut Courtois", "Real Madrid", "GK", 6.0, "ffs_premium", ""),
    ("Gianluigi Donnarumma", "Man City", "GK", 6.0, "ffs_premium", "club from FFS watchlist"),
    ("Alisson Becker", "Liverpool", "GK", 6.0, "ffs_premium", ""),
    ("Matvey Safonov", "PSG", "GK", 5.5, "ffs_premium", "club not stated in article"),
    ("Manuel Neuer", "Bayern", "GK", 5.5, "ffs_premium", ""),
    ("Jan Oblak", "Atleti", "GK", 5.5, "ffs_premium", ""),
    ("Yann Sommer", "Club Brugge", "GK", 4.5, "uefa_enablers", "club as stated by source"),
    ("Josep Martínez", "Inter", "GK", 4.5, "uefa_enablers", ""),
    ("Ivan Provedel", "Inter", "GK", 4.5, "uefa_enablers", ""),
    ("Thomas Strakosha", "AEK Athens", "GK", 4.5, "uefa_enablers", ""),
    ("Fabian Bredlow", "Stuttgart", "GK", 4.0, "uefa_enablers", "covering for injured No.1"),
    ("Zion Suzuki", "Aston Villa", "GK", 4.0, "uefa_enablers", ""),
    ("Marco Bizot", "Club Brugge", "GK", 4.0, "uefa_enablers", "club as stated by source"),
    # premium / mid-price defenders
    ("Willian Pacho", "PSG", "DEF", 6.0, "ffs_premium", ""),
    ("Achraf Hakimi", "PSG", "DEF", 6.0, "ffs_premium", "takes set plays"),
    ("Marquinhos", "PSG", "DEF", 5.0, "ffs_premium", ""),
    ("Dayot Upamecano", "Bayern", "DEF", 5.5, "ffs_premium", "12 recoveries in MD1"),
    ("Gabriel Magalhães", "Arsenal", "DEF", 6.0, "ffs_md2", ""),
    ("Virgil van Dijk", "Liverpool", "DEF", 6.0, "ffs_md2", ""),
    ("Federico Dimarco", "Inter", "DEF", 5.0, "ffs_md2", "fully fit per FFS 11 Oct"),
    ("Harry Maguire", "Man United", "DEF", 5.0, "ffs_md1", ""),
    # midfielders
    ("Khvicha Kvaratskhelia", "PSG", "MID", 10.0, "uefa_mid", ""),
    ("Lamine Yamal", "Barcelona", "MID", 10.0, "uefa_mid", "on penalties while Raphinha is out (FFS 11 Oct)"),
    ("Raphinha", "Barcelona", "MID", 9.5, "uefa_mid", "injured (FFS 11 Oct)"),
    ("Vinícius Júnior", "Real Madrid", "MID", 9.5, "uefa_mid", ""),
    ("Bruno Fernandes", "Man United", "MID", 9.0, "uefa_mid", "penalty taker"),
    ("Michael Olise", "Bayern", "MID", 9.0, "uefa_mid", ""),
    ("Jude Bellingham", "Real Madrid", "MID", 8.5, "uefa_mid", ""),
    ("Bukayo Saka", "Arsenal", "MID", 8.5, "uefa_mid", ""),
    ("Anthony Gordon", "Barcelona", "MID", 8.0, "uefa_mid", "new signing"),
    ("Vitinha", "PSG", "MID", 7.5, "uefa_mid", ""),
    ("Dominik Szoboszlai", "Liverpool", "MID", 7.0, "uefa_mid", ""),
    ("Rayan Cherki", "Man City", "MID", 7.0, "uefa_mid", ""),
    ("Kevin De Bruyne", "Napoli", "MID", 7.0, "ffs_md2", ""),
    ("Nico Paz", "Como", "MID", 7.0, "ffs_md2", ""),
    ("Hans Vanaken", "Club Brugge", "MID", 6.5, "uefa_mid", ""),
    ("Florian Thauvin", "Lens", "MID", 6.5, "uefa_mid", ""),
    ("Anis Hadj Moussa", "Feyenoord", "MID", 6.5, "uefa_mid", ""),
    ("Ismael Saibari", "Bayern", "MID", 6.5, "uefa_enablers", ""),
    ("João Neves", "PSG", "MID", 6.0, "uefa_mid", ""),
    ("Martin Baturina", "Como", "MID", 5.5, "uefa_mid", ""),
    ("Maxi Araújo", "Sporting", "MID", 5.5, "uefa_mid", ""),
    ("Zlatko Tripić", "Viking", "MID", 5.5, "uefa_mid", ""),
    ("Răzvan Marin", "AEK Athens", "MID", 5.5, "ffs_mid55", "set pieces and penalties"),
    ("Guus Til", "PSV", "MID", 5.5, "ffs_mid55", ""),
    ("Konstantinos Karetsas", "Dortmund", "MID", 5.5, "ffs_mid55", ""),
    ("Chris Führich", "Stuttgart", "MID", 5.5, "ffs_mid55", ""),
    ("Álex Baena", "Atleti", "MID", 5.5, "ffs_mid55", "club not stated in article"),
    ("Lovro Majer", "AEK Athens", "MID", 5.0, "ffs_mid55", ""),
    ("Aleksandar Pavlović", "Bayern", "MID", 5.0, "uefa_mid", "club not stated in article"),
    ("Rodrigo Zalazar", "Sporting", "MID", 4.5, "uefa_enablers", "takes corners"),
    ("Ridle Baku", "Leipzig", "MID", 4.5, "uefa_enablers", ""),
    # forwards
    ("Harry Kane", "Bayern", "FWD", 11.0, "uefa_fwd", "penalty taker"),
    ("Kylian Mbappé", "Real Madrid", "FWD", 11.0, "uefa_fwd", "penalty taker"),
    ("Erling Haaland", "Man City", "FWD", 11.0, "uefa_fwd", "penalty taker"),
    ("Ousmane Dembélé", "PSG", "FWD", 10.0, "uefa_fwd", ""),
    ("Julián Álvarez", "Atleti", "FWD", 10.0, "ffs_premium", "club not stated in article"),
    ("Lautaro Martínez", "Inter", "FWD", 9.5, "uefa_fwd", ""),
    ("Viktor Gyökeres", "Arsenal", "FWD", 8.5, "uefa_fwd", ""),
    ("Serhou Guirassy", "Dortmund", "FWD", 8.5, "uefa_fwd", ""),
    ("Victor Osimhen", "Galatasaray", "FWD", 8.0, "uefa_fwd", ""),
    ("Kai Havertz", "Arsenal", "FWD", 8.0, "uefa_fwd", ""),
    ("Rasmus Højlund", "Napoli", "FWD", 8.0, "uefa_fwd", ""),
    ("Alexander Isak", "Liverpool", "FWD", 7.5, "uefa_fwd", "club from FFS watchlist"),
    ("Ferran Torres", "Barcelona", "FWD", 7.0, "uefa_fwd", "club not stated in article"),
    ("Karim Adeyemi", "Dortmund", "FWD", 7.0, "uefa_fwd", "club not stated in article"),
    ("Luis Suárez", "Sporting", "FWD", 7.0, "uefa_fwd", ""),
    ("Nicolò Tresoldi", "Club Brugge", "FWD", 7.0, "uefa_fwd", "club not stated in article"),
    ("Bryan Mbeumo", "Man United", "FWD", 7.0, "uefa_fwd", "club not stated in article"),
    ("Alexander Sørloth", "Atleti", "FWD", 7.0, "uefa_fwd", "club not stated in article"),
    ("Deniz Undav", "Stuttgart", "FWD", 7.0, "ffs_fwd", ""),
    ("Donyell Malen", "Roma", "FWD", 6.5, "uefa_fwd", ""),
    ("Olivier Giroud", "Lille", "FWD", 6.5, "uefa_fwd", ""),
    ("Benjamin Šeško", "Man United", "FWD", 6.5, "uefa_fwd", ""),
    ("Francesco Pio Esposito", "Inter", "FWD", 6.5, "uefa_enablers", "uefa_fwd lists €5.0m; enablers article (later) €6.5m"),
    ("Barnabás Varga", "AEK Athens", "FWD", 6.5, "ffs_fwd", ""),
    ("Luka Jović", "AEK Athens", "FWD", 6.5, "ffs_fwd", ""),
    ("Anastasios Douvikas", "Como", "FWD", 6.5, "ffs_fwd", ""),
    ("Moise Kean", "Como", "FWD", 6.5, "ffs_fwd", ""),
    ("Ricardo Pepi", "PSV", "FWD", 6.0, "ffs_fwd", ""),
    ("Cucho Hernández", "Real Betis", "FWD", 6.0, "ffs_fwd", ""),
    ("Ermedin Demirović", "Stuttgart", "FWD", 6.0, "ffs_fwd", ""),
    ("Maximilian Beier", "Dortmund", "FWD", 6.0, "ffs_fwd", ""),
    ("Vedat Muriqi", "Fenerbahce", "FWD", 6.0, "uefa_enablers", ""),
    ("Mamadou Diakhon", "Club Brugge", "FWD", 5.5, "ffs_fwd", ""),
    ("Troy Parrott", "Real Betis", "FWD", 5.0, "ffs_fwd", ""),
    ("Dženan Pejčinović", "Stuttgart", "FWD", 5.0, "ffs_fwd", ""),
    ("Andreas Helmersen", "Bodo/Glimt", "FWD", 5.0, "uefa_enablers", ""),
]


def main() -> None:
    rows = []
    for block, price in ((DEF_400, 4.0), (DEF_450, 4.5)):
        for line in block.strip().splitlines():
            club, names = line.split(":", 1)
            for name in names.split(";"):
                rows.append(dict(name=name.strip(), team=CLUB[club.strip()], pos="DEF", price=price,
                                 source="ffs_def200", as_of="2026-09-01", note=""))
    for name, club, pos, price, src, note in OTHERS:
        as_of = {"ffs_md2": "2026-10-11", "ffs_md1": "2026-09-08", "ffs_premium": "2026-09-07",
                 "ffs_mid55": "2026-09-07", "ffs_fwd": "2026-09-08", "uefa_fwd": "2026-09-01",
                 "uefa_mid": "2026-09-01", "uefa_enablers": "2026-09-02"}[src]
        rows.append(dict(name=name, team=CLUB[club], pos=pos, price=price, source=src, as_of=as_of, note=note))
    seen, out = set(), []
    for r in rows:          # de-duplicate by (name, team); later sources win
        key = (r["name"].lower(), r["team"])
        if key in seen:
            out = [o for o in out if (o["name"].lower(), o["team"]) != key]
        seen.add(key)
        out.append(r)
    dest = ROOT / "data" / "seed" / "players_2026_27.csv"
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "team", "pos", "price", "source", "as_of", "note"])
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} players -> {dest}")


if __name__ == "__main__":
    main()
