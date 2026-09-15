import ROOT
import numpy as np

ROOT.gROOT.SetBatch(True)

NOME_ARQUIVO = "AO2D.root"
PT_MIN = 0
NSIGMA_MAX = 1.0
N_BINS_P_CURVA = 40
P_MIN_CURVA, P_MAX_CURVA = 0.12, 10.0
MIN_PONTOS_POR_BIN = 15

HIPOTESES_PID = {
    "electron": ("O2pidtpcel", "fTPCNSigmaStoreEl"),
    "muon": ("O2pidtpcmu", "fTPCNSigmaStoreMu"),
    "pion": ("O2pidtpcpi", "fTPCNSigmaStorePi"),
    "kaon": ("O2pidtpcka", "fTPCNSigmaStoreKa"),
    "proton": ("O2pidtpcpr", "fTPCNSigmaStorePr"),
    "deuteron": ("O2pidtpcde", "fTPCNSigmaStoreDe"),
    "triton": ("O2pidtpctr", "fTPCNSigmaStoreTr"),
    "helium3": ("O2pidtpche", "fTPCNSigmaStoreHe"),
    "helium4": ("O2pidtpcal", "fTPCNSigmaStoreAl"),
}

NOMES_PARTICULAS = {
    "electron": "eletron",
    "muon": "muon",
    "pion": "pion",
    "kaon": "kaon",
    "proton": "proton",
    "deuteron": "deuteron",
    "triton": "tritio",
    "helium3": "helio-3",
    "helium4": "helio-4",
}

def listar_diretorios(arquivo):
    nomes = []

    for key in arquivo.GetListOfKeys():
        obj = arquivo.Get(key.GetName())

        if obj.InheritsFrom("TDirectoryFile"):
            nomes.append(key.GetName())

    return nomes

def processar_diretorio(arquivo, nome_diretorio):
    diretorio = arquivo.Get(nome_diretorio)
    tree_track = diretorio.Get("O2track")
    tree_trackextra = diretorio.Get("O2trackextra_002")

    if not tree_track:
        print("   [aviso] O2track não encontrado.")
        return None

    if not tree_trackextra:
        print("   [aviso] O2trackextra_002 não encontrado.")
        return None

    n_track = tree_track.GetEntries()
    trees_pid = {}

    for especie, (nome_tree, nome_branch) in HIPOTESES_PID.items():
        tree = diretorio.Get(nome_tree)

        if not tree:
            continue

        if tree.GetEntries() != n_track:
            print(f"   [aviso] {nome_tree} tem número de entradas diferente de O2track.")
            continue

        trees_pid[especie] = (tree, nome_branch)

    if len(trees_pid) == 0:
        print("   [aviso] Nenhuma árvore de PID TPC encontrada.")
        return None

    print(f"   -> hipóteses de PID encontradas: {len(trees_pid)}")

    tree_track.AddFriend(tree_trackextra, "trackextra")

    for especie, (tree, _) in trees_pid.items():
        tree_track.AddFriend(tree, f"pid_{especie}")

    rdf = ROOT.RDataFrame(tree_track)

    rdf = (
        rdf
        .Define("pt_", "1.0/fabs(fSigned1Pt)")
        .Define("eta_", "asinh(fTgl)")
        .Define("p_", "pt_ * cosh(eta_)")
        .Filter(f"pt_ > {PT_MIN} && fTPCSignal > 0")
    )

    colunas = ["pt_", "eta_", "p_", "fTPCSignal"]

    for especie, (_, nome_branch) in trees_pid.items():
        nome_coluna = f"nsigma_{especie}_"
        rdf = rdf.Define(nome_coluna, f"(float)pid_{especie}.{nome_branch}")
        colunas.append(nome_coluna)

    dados = rdf.AsNumpy(columns=colunas)

    return {chave: np.asarray(valor) for chave, valor in dados.items()}

def identificar_particulas(dados):
    especies = [
        especie
        for especie in HIPOTESES_PID
        if f"nsigma_{especie}_" in dados
    ]

    n = len(dados["p_"])
    identificacao = np.full(n, -1, dtype=int)
    menor_nsigma = np.full(n, np.inf)

    for indice, especie in enumerate(especies):
        nsigma = np.abs(dados[f"nsigma_{especie}_"])
        mascara = nsigma < menor_nsigma
        menor_nsigma[mascara] = nsigma[mascara]
        identificacao[mascara] = indice

    identificacao[menor_nsigma > NSIGMA_MAX] = -1

    return identificacao, menor_nsigma, especies

def curva_via_nsigma(p, dedx, nsigma):
    limpo = np.abs(nsigma) < NSIGMA_MAX

    bordas = np.logspace(
        np.log10(P_MIN_CURVA),
        np.log10(P_MAX_CURVA),
        N_BINS_P_CURVA + 1
    )

    centros = []
    medianas = []

    for i in range(N_BINS_P_CURVA):
        na_faixa = limpo & (p >= bordas[i]) & (p < bordas[i + 1])

        if na_faixa.sum() >= MIN_PONTOS_POR_BIN:
            centros.append(np.sqrt(bordas[i] * bordas[i + 1]))
            medianas.append(np.median(dedx[na_faixa]))

    return np.array(centros), np.array(medianas)

def criar_tgraph(x_vals, y_vals, cor, nome):
    g = ROOT.TGraph(
        len(x_vals),
        x_vals.astype("float64"),
        y_vals.astype("float64")
    )

    g.SetName(nome)
    g.SetLineColor(cor)
    g.SetLineWidth(2)
    g.SetLineStyle(1)

    return g

def criar_th1_do_numpy(valores, nome, titulo, xmin, xmax, nbins=100):
    counts, edges = np.histogram(valores, bins=nbins, range=(xmin, xmax))

    h = ROOT.TH1F(nome, titulo, nbins, xmin, xmax)

    for i in range(nbins):
        h.SetBinContent(i + 1, counts[i])

    return h

def criar_th2_do_numpy(x_vals, y_vals, nome, titulo, x_range, y_range, nbins=200, log_x=True):
    if log_x:
        x_edges = np.logspace(
            np.log10(x_range[0]),
            np.log10(x_range[1]),
            nbins + 1
        )
    else:
        x_edges = np.linspace(x_range[0], x_range[1], nbins + 1)

    y_edges = np.linspace(y_range[0], y_range[1], nbins + 1)

    counts, _, _ = np.histogram2d(
        x_vals,
        y_vals,
        bins=[x_edges, y_edges]
    )

    h = ROOT.TH2F(
        nome,
        titulo,
        nbins,
        x_edges,
        nbins,
        y_edges
    )

    for i in range(nbins):
        for j in range(nbins):
            if counts[i, j] > 0:
                h.SetBinContent(i + 1, j + 1, counts[i, j])

    return h

def main():
    print("1 - abrindo arquivo...")

    arquivo = ROOT.TFile.Open(NOME_ARQUIVO)

    if not arquivo or arquivo.IsZombie():
        print("Erro ao abrir o arquivo!")
        return

    nomes_diretorios = listar_diretorios(arquivo)
    print(f"2 - diretórios encontrados: {len(nomes_diretorios)}")

    acumulado = {
        "pt_": [],
        "eta_": [],
        "p_": [],
        "fTPCSignal": []
    }

    for especie in HIPOTESES_PID:
        acumulado[f"nsigma_{especie}_"] = []

    for i, nome_dir in enumerate(nomes_diretorios):
        print(f"3.{i+1} - processando {nome_dir}...")

        resultado = processar_diretorio(arquivo, nome_dir)

        if resultado is None:
            continue

        for chave in resultado:
            if chave in acumulado:
                acumulado[chave].append(resultado[chave])

        print(f"      -> {len(resultado['pt_'])} tracks")

    dados = {}

    for chave, valores in acumulado.items():
        if len(valores) > 0:
            dados[chave] = np.concatenate(valores)

    pt = dados["pt_"]
    eta = dados["eta_"]
    p = dados["p_"]
    dedx = dados["fTPCSignal"]

    print(f"\n4 - total combinado: {len(pt)} tracks")

    print("\n5 - identificando partículas pelo menor |nσ|...")

    identificacao, menor_nsigma, especies = identificar_particulas(dados)

    print("\nIDENTIFICACAO:")

    for indice, especie in enumerate(especies):
        n = np.sum(identificacao == indice)
        print(f"   {NOMES_PARTICULAS[especie]:10s}: {n} tracks")

    n_nao_identificadas = np.sum(identificacao == -1)
    print(f"   não identificadas: {n_nao_identificadas} tracks")

    print("\n6 - calculando curvas via nσ...")

    curvas = {}

    for indice, especie in enumerate(especies):
        mascara = identificacao == indice

        p_curva, dedx_curva = curva_via_nsigma(
            p[mascara],
            dedx[mascara],
            dados[f"nsigma_{especie}_"][mascara]
        )

        curvas[especie] = (p_curva, dedx_curva)

        print(f"   {NOMES_PARTICULAS[especie]:10s}: {len(p_curva)} bins")

    h_pt = criar_th1_do_numpy(
        pt,
        "h_pt",
        f"Distribuicao de p_{{T}};p_{{T}} (GeV/c);Contagens",
        0,
        20
    )

    c1 = ROOT.TCanvas("c1", "pT", 800, 600)
    c1.SetLogy()
    h_pt.Draw()
    c1.SaveAs("tpc_01_distribuicao_pt.png")

    h_dedx = criar_th1_do_numpy(
        dedx,
        "h_dedx",
        "Distribuicao de dE/dx;dE/dx (u.a.);Contagens",
        0,
        200
    )

    c2 = ROOT.TCanvas("c2", "dEdx", 800, 600)
    h_dedx.Draw()
    c2.SaveAs("tpc_02_distribuicao_dedx.png")

    h_2d = criar_th2_do_numpy(
        p,
        dedx,
        "h_dedx_vs_p",
        "TPC: dE/dx vs p;p (GeV/c);dE/dx (u.a.)",
        x_range=(0.1, 10),
        y_range=(0, 200),
        nbins=200,
        log_x=True
    )

    c3 = ROOT.TCanvas("c3", "dEdx vs p", 900, 650)
    c3.SetLogx()
    c3.SetLogz()
    h_2d.Draw("COLZ")
    c3.SaveAs("tpc_03_dedx_vs_p.png")

    c4 = ROOT.TCanvas("c4", "dEdx vs p com PID", 900, 650)
    c4.SetLogx()
    c4.SetLogz()
    h_2d.Draw("COLZ")

    cores = [
        ROOT.kRed,
        ROOT.kBlue,
        ROOT.kGreen + 2,
        ROOT.kMagenta,
        ROOT.kCyan + 2,
        ROOT.kOrange + 7,
        ROOT.kViolet,
        ROOT.kPink + 1,
        ROOT.kGray + 2
    ]

    legenda = ROOT.TLegend(0.15, 0.65, 0.25, 0.88) # esquerda, baixo, direita, cima
    legenda.SetTextSize(0.025)
    graficos = []

    for i, especie in enumerate(especies):
        p_curva, dedx_curva = curvas[especie]

        if len(p_curva) == 0:
            continue

        g = criar_tgraph(
            p_curva,
            dedx_curva,
            cores[i],
            f"g_{especie}"
        )

        g.Draw("L SAME")

        legenda.AddEntry(
            g,
            NOMES_PARTICULAS[especie],
            "l"
        )

        graficos.append(g)

    legenda.Draw()
    c4.SaveAs("tpc_04_dedx_vs_p_todas_particulas.png")

    dados["identificacao"] = identificacao
    dados["menor_nsigma"] = menor_nsigma

    np.savez("tpc_dados_com_pid.npz", **dados)

    print("\n7 - dados salvos em tpc_dados_com_pid.npz")
    print("\nFIM -- PID TPC com todas as hipóteses processado com sucesso.")

if __name__ == "__main__":
    main()
