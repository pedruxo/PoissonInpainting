"""
Projeto de Inpainting com Métodos de Interpolação e Gauss-Seidel Red-Black
Este projeto implementa técnicas de inpainting em imagens sintéticas,
incluindo interpolação bilinear, interpolação Lagrange 2D e um método de Gauss-Seidel Red-Black.
"""

import math
from PIL import Image


def cria_imagem_sintetica(largura, altura):
    """
    Cria uma imagem sintética em escala de cinza como um objeto imagem PIL.
    Simula um gradiente para fins de teste.
    """
    imagem = Image.new("L", (largura, altura))
    pixels = imagem.load()

    for r in range(altura):
        for c in range(largura):
            valor_pixel = int(255 * (r / altura + c / largura) / 2)
            pixels[c, r] = valor_pixel
    return imagem


def imagem_pil_para_lista(imagem):
    """
    Converte uma imagem PIL em uma lista de listas de pixels.
    """
    largura, altura = imagem.size
    lista_pixels = []
    for r in range(altura):
        linha = []
        for c in range(largura):
            linha.append(imagem.getpixel((c, r)))
        lista_pixels.append(linha)
    return lista_pixels


def lista_para_imagem_pil(lista_pixels):
    """
    Converte uma lista de listas de pixels em uma imagem PIL.
    """
    altura = len(lista_pixels)
    largura = len(lista_pixels[0])
    imagem = Image.new("L", (largura, altura))
    pixels = imagem.load()

    for r in range(altura):
        for c in range(largura):
            pixels[c, r] = int(round(lista_pixels[r][c]))
    return imagem


def salva_imagem(lista_pixels, caminho):
    """
    Converte uma imagem PIL em um arquivo e salva no caminho especificado.
    """
    imagem_pil = lista_para_imagem_pil(lista_pixels)
    imagem_pil.save(caminho)
    print(f"Imagem salva em: {caminho}")


def cria_mascara(formato_img, tipo_mascara="rectangle", params=None):
    """
    Cria uma imagem booleana para a imagem
    True indica pixel faltante e False indica pixel presente.
    """
    altura, largura = formato_img
    mascara = [[False for _ in range(largura)] for _ in range(altura)]

    if tipo_mascara == "rectangle":
        if params is None:
            linha_comeco = altura // 4
            linha_fim = altura * 3 // 4
            coluna_comeco = largura // 4
            coluna_fim = largura * 3 // 4
        else:
            linha_comeco, linha_fim, coluna_comeco, coluna_fim = params

        for r in range(linha_comeco, linha_fim):
            for c in range(coluna_comeco, coluna_fim):
                mascara[r][c] = True
    elif tipo_mascara == "circle":
        if params is None:
            linha_centro, coluna_centro, raio = (
                altura // 2,
                largura // 2,
                min(altura, largura) // 4,
            )
        else:
            linha_centro, coluna_centro, raio = params

        for r in range(altura):
            for c in range(largura):
                if (r - linha_centro) ** 2 + (c - coluna_centro) ** 2 <= raio**2:
                    mascara[r][c] = True

    else:
        raise ValueError("Tipo de máscara desconhecido. Use 'rectangle' ou 'circle'.")
    return mascara


def aplica_mascara_para_lista_imagem(lista_pixels, mascara):
    """
    Cria uma cópia da lista de pixels e aplica a máscara,
    definindo os pixels mascarados como 0
    Retorna a liamgem mascarada em formato de lista de listas.
    """
    altura = len(lista_pixels)
    largura = len(lista_pixels[0])
    imagem_mascarada = [
        [lista_pixels[r][c] for c in range(largura)] for r in range(altura)
    ]

    for r in range(altura):
        for c in range(largura):
            if mascara[r][c]:
                imagem_mascarada[r][c] = 0
    return imagem_mascarada


def copia_lista_imagem(lista_pixels):
    """
    Cria uma cópia da lista de pixels.
    """
    return [row[:] for row in lista_pixels]


# ---   Métricas de Qualidade ---
def calcula_mse(original, reconstruida):
    """
    Calcula o Erro Quadrático Médio (MSE).
    """
    altura = len(original)
    largura = len(original[0])
    mse = 0.0

    for r in range(altura):
        for c in range(largura):
            mse += (original[r][c] - reconstruida[r][c]) ** 2
    return mse / (altura * largura)


def calcula_psnr(original, reconstruida):
    """
    Calcula o Pico de Sinal sobre Ruído (PSNR).
    MAX_I é 255 para imagens de 8 bits.
    """
    mse = calcula_mse(original, reconstruida)
    if mse == 0:
        return float("inf")
    max_i = 255.0
    psnr = 10 * math.log10(max_i**2 / mse)
    return psnr


def calcula_ssim(original, reconstruida, k1=0.01, k2=0.03, l=255, window_size=11):
    """
    Calcula o Índice de Similaridade Estrutural (SSIM).
    Esta é uma implementação simplificada e não otimizada.
    """
    altura = len(original)
    largura = len(original[0])

    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2

    half_window = window_size // 2

    total_ssim = 0.0
    count_windows = 0

    for r in range(half_window, altura - half_window):
        for c in range(half_window, largura - half_window):
            window_original = []
            window_reconstruida = []

            for wr in range(r - half_window, r + half_window + 1):
                for wc in range(c - half_window, c + half_window + 1):
                    window_original.append(original[wr][wc])
                    window_reconstruida.append(reconstruida[wr][wc])

            mu_x = sum(window_original) / len(window_original)
            mu_y = sum(window_reconstruida) / len(window_reconstruida)

            sigma_x = sum([(p - mu_x) ** 2 for p in window_original]) / len(
                window_original
            )
            sigma_y = sum([(p - mu_y) ** 2 for p in window_reconstruida]) / len(
                window_reconstruida
            )

            sigma_xy = sum(
                [
                    (window_original[i] - mu_x) * (window_reconstruida[i] - mu_y)
                    for i in range(len(window_original))
                ]
            ) / len(window_original)

            numerador = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            denominador = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

            if denominador == 0:
                valor_ssim = 1.0
            else:
                valor_ssim = numerador / denominador

            total_ssim += valor_ssim
            count_windows += 1

    return total_ssim / count_windows if count_windows > 0 else 0.0


# --- Métodos de Impainting ---
def gauss_seidel_red_black(lista_pixels, mascara, iteracoes=800, tolerancia=0.01):
    """
    Implementa o Impainting Poissoniano usando Gauss-Seidel com ordenação Red-Black.
    Opera em listas de listas de pixels.
    """
    altura = len(lista_pixels)
    largura = len(lista_pixels[0])
    imagem_reconstruida = copia_lista_imagem(lista_pixels)

    pixels_red = []
    pixels_black = []
    for r in range(altura):
        for c in range(largura):
            if mascara[r][c]:
                if (r + c) % 2 == 0:
                    pixels_red.append((r, c))
                else:
                    pixels_black.append((r, c))

    print(f"Iniciando Gauss-Seidel Red-Black com {iteracoes} iterações...")
    for it_count in range(iteracoes):
        erro_max = 0.0

        for r, c in pixels_red:
            soma_vizinhos = 0.0
            count_vizinhos = 0
            erro = 0.0

            if r > 0:
                soma_vizinhos += imagem_reconstruida[r - 1][c]
                count_vizinhos += 1
            if r < altura - 1:
                soma_vizinhos += imagem_reconstruida[r + 1][c]
                count_vizinhos += 1
            if c > 0:
                soma_vizinhos += imagem_reconstruida[r][c - 1]
                count_vizinhos += 1
            if c < largura - 1:
                soma_vizinhos += imagem_reconstruida[r][c + 1]
                count_vizinhos += 1

            if count_vizinhos > 0:
                novo_valor = int(round(soma_vizinhos / count_vizinhos))
                erro += abs(imagem_reconstruida[r][c] - novo_valor)

                if erro > erro_max:
                    erro_max = erro
                imagem_reconstruida[r][c] = novo_valor

        for r, c in pixels_black:
            soma_vizinhos = 0.0
            count_vizinhos = 0

            if r > 0:
                soma_vizinhos += imagem_reconstruida[r - 1][c]
                count_vizinhos += 1
            if r < altura - 1:
                soma_vizinhos += imagem_reconstruida[r + 1][c]
                count_vizinhos += 1
            if c > 0:
                soma_vizinhos += imagem_reconstruida[r][c - 1]
                count_vizinhos += 1
            if c < largura - 1:
                soma_vizinhos += imagem_reconstruida[r][c + 1]
                count_vizinhos += 1

            if count_vizinhos > 0:
                novo_valor = int(round(soma_vizinhos / count_vizinhos))
                erro += abs(imagem_reconstruida[r][c] - novo_valor)
                if erro > erro_max:
                    erro_max = erro
                imagem_reconstruida[r][c] = novo_valor

        if erro_max < tolerancia:
            print(f"Convergência alcançada após {it_count + 1} iterações.")
            break
        if (it_count + 1) % 100 == 0:
            print(f"Iteração {it_count + 1}/{iteracoes}, Erro máximo: {erro_max:.4f}")

    print("Gauss-Seidel Red-Black concluído.")
    return imagem_reconstruida


def interpolacao_bilinear(lista_pixels, mascara):
    """
    Preenche furos usando interpolação bilinear.
    Para cada pixel mascarado, encontra 4 vizinhos conhecidos e interpola.
    Opera em listas de listas de pixels.
    """
    altura = len(lista_pixels)
    largura = len(lista_pixels[0])
    imagem_reconstruida = copia_lista_imagem(lista_pixels)

    print("Iniciando interpolação bilinear...")
    for r in range(altura):
        for c in range(largura):
            if mascara[r][c]:
                x1, y1, x2, y2 = -1, -1, -1, -1
                q11, q12, q21, q22 = 0, 0, 0, 0

                for coluna_esq in range(c - 1, -1, -1):
                    if not mascara[r][coluna_esq]:
                        x1 = coluna_esq
                        break
                for coluna_dir in range(c + 1, largura):
                    if not mascara[r][coluna_dir]:
                        x2 = coluna_dir
                        break
                for linha_acima in range(r - 1, -1, -1):
                    if not mascara[linha_acima][c]:
                        y1 = linha_acima
                        break
                for linha_abaixo in range(r + 1, altura):
                    if not mascara[linha_abaixo][c]:
                        y2 = linha_abaixo
                        break

                if x1 != -1 and x2 != -1 and y1 != -1 and y2 != -1:
                    q11 = imagem_reconstruida[y1][x1]
                    q12 = imagem_reconstruida[y1][x2]
                    q21 = imagem_reconstruida[y2][x1]
                    q22 = imagem_reconstruida[y2][x2]

                    dx = (c - x1) / (x2 - x1)
                    dy = (r - y1) / (y2 - y1)

                    r1 = q11 * (1 - dx) + q12 * dx
                    r2 = q21 * (1 - dx) + q22 * dx

                    valor_pixel = int(round(r1 * (1 - dy) + r2 * dy))
                    imagem_reconstruida[r][c] = valor_pixel
                else:
                    pass  # Não há vizinhos suficientes para interpolar completamente
    print("Interpolação bilinear concluída.")
    return imagem_reconstruida


def interpolacao_lagrange_2d(lista_pixels, mascara, vizinhos_tam=3):
    """
    Preenche furos usando interpolação em lagrange 2D.
    Opera em lstas de listas de pixels.
    """

    altura = len(lista_pixels)
    largura = len(lista_pixels[0])
    imagem_reconstruida = copia_lista_imagem(lista_pixels)

    print(
        f"Iniciando interpolação Lagrange 2D com vizinhança de tamanho {vizinhos_tam}..."
    )

    def lagrange_aux(x, pontos, k):
        """Calcula o k-ésimo termo de Lagrange."""

        resultado = 1.0
        for j, p_j in enumerate(pontos):  # j -> índice, p_j -> valor do ponto
            if j != k:
                resultado *= (x - p_j[0]) / (pontos[k][0] - p_j[0])
        return resultado

    for r in range(altura):
        for c in range(largura):
            if mascara[r][c]:
                pontos_x = []
                pontos_y = []

                for col_index in range(
                    max(0, c - vizinhos_tam), min(largura, c + vizinhos_tam + 1)
                ):
                    if not mascara[r][col_index]:
                        pontos_x.append((col_index, imagem_reconstruida[r][col_index]))
                for linha_index in range(
                    max(0, r - vizinhos_tam), min(altura, r + vizinhos_tam + 1)
                ):
                    if not mascara[linha_index][c]:
                        pontos_y.append(
                            (linha_index, imagem_reconstruida[linha_index][c])
                        )

                if len(pontos_x) >= 2 and len(pontos_y) >= 2:
                    pontos_grade = []
                    coords_x = sorted(list(set([p[0] for p in pontos_x])))
                    coords_y = sorted(list(set([p[0] for p in pontos_y])))

                    if len(coords_x) < 2 or len(coords_y) < 2:
                        continue

                    for valor_coord_y in coords_y:
                        for valor_coord_x in coords_x:
                            if not mascara[valor_coord_y][valor_coord_x]:
                                pontos_grade.append(
                                    (
                                        valor_coord_x,
                                        valor_coord_y,
                                        imagem_reconstruida[valor_coord_y][
                                            valor_coord_x
                                        ],
                                    )
                                )

                    if len(pontos_grade) < 4:
                        continue

                    valor_interpolado = 0.0
                    for valor_x, valor_y, valor in pontos_grade:
                        x_index = coords_x.index(valor_x)
                        y_index = coords_y.index(valor_y)

                        x_base = lagrange_aux(c, pontos_x, x_index)
                        y_base = lagrange_aux(r, pontos_y, y_index)
                        valor_interpolado += valor * x_base * y_base

                    imagem_reconstruida[r][c] = int(round(valor_interpolado))
                else:
                    pass

    print("Interpolação Lagrange 2D concluída.")
    return imagem_reconstruida


# --- Função Principal ---


def run_projeto():
    """
    Função principal para executar o projeto de inpainting.
    """
    largura_img = 256
    altura_img = 256

    # 1. Cria a imagem original (Imagem PIL)
    img_pil_original = cria_imagem_sintetica(largura_img, altura_img)
    lista_img_original = imagem_pil_para_lista(img_pil_original)

    # 2. Cria a máscara
    mascara_params = (
        # altura_img // 4,
        # altura_img * 3 // 4,
        # largura_img // 4,
        # largura_img * 3 // 4,
        64,
        192,
        64,
        192,
    )
    mascara = cria_mascara(
        (largura_img, altura_img), tipo_mascara="rectangle", params=mascara_params
    )

    # 3. Aplica a máscara à imagem original
    # Esta é a etapa onde ocorre o *furo* na imagem original
    img_mascarada_teste = aplica_mascara_para_lista_imagem(lista_img_original, mascara)

    # 4. Salva a imagem original e a mascarada
    salva_imagem(lista_img_original, "imagem_original.png")
    salva_imagem(img_mascarada_teste, "imagem_mascarada.png")

    result = {}

    print(
        "\n--- Executando Impainting Poissoniano com o método Gauss-Seidel Red-Black ---"
    )
    gs_rb_reconstruida = gauss_seidel_red_black(
        img_mascarada_teste, mascara, iteracoes=2000, tolerancia=0.1
    )
    result["Gauss-Seidel Red-Black"] = gs_rb_reconstruida
    salva_imagem(gs_rb_reconstruida, "imagem_reconstruida_gs_rb.png")

    # --- Comparações com outros métodos ---

    print("\n--- Executando Interpolação Bilinear ---")
    bilinear_reconstruida = interpolacao_bilinear(img_mascarada_teste, mascara)
    result["Bilinear"] = bilinear_reconstruida
    salva_imagem(bilinear_reconstruida, "imagem_reconstruida_bilinear.png")

    print("\n--- Executando Interpolação Lagrange 2D ---")
    lagrange_reconstruida = interpolacao_lagrange_2d(
        img_mascarada_teste, mascara, vizinhos_tam=2
    )
    result["Lagrange 2D"] = lagrange_reconstruida
    salva_imagem(lagrange_reconstruida, "imagem_reconstruida_lagrange.png")

    # --- Análise de qualidade ---
    print("\n--- Análise de Qualidade ---")
    for metodo, img_reconstruida in result.items():
        psnr = calcula_psnr(lista_img_original, img_reconstruida)
        ssim = calcula_ssim(lista_img_original, img_reconstruida)

        print(f"\nMétodo: {metodo}")
        print(f" PSNR: {psnr:.2f} dB")
        print(f" SSIM: {ssim:.4f}")

    print("\n--- Resultados e Saída de Texto simplificada ---")
    print(
        "As imagens original, mascarada e reconstruídas foram salvas como arquivo .png"
    )
    print("\nImagem original:")
    for r in range(5):
        print([lista_img_original[r][c] for c in range(10)])

    print("\nImagem mascarada:")
    for r in range(5):
        print([img_mascarada_teste[r][c] for c in range(10)])

    print("\nImagem reconstruída Gauss-Seidel Red-Black:")
    for r in range(5):
        print([gs_rb_reconstruida[r][c] for c in range(10)])

    print("\nImagem reconstruída Bilinear:")
    for r in range(5):
        print([bilinear_reconstruida[r][c] for c in range(10)])

    print("\nImagem reconstruída Lagrange 2D:")
    for r in range(5):
        print([lagrange_reconstruida[r][c] for c in range(10)])


# Execução do projeto
if __name__ == "__main__":
    run_projeto()
