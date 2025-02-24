# Utilized ChatGPT
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import quad


# Define the standard log-normal PDF
def ln_PDF(D, mu, sigma):
    if D == 0:
        return 0
    return 1 / (D * sigma * math.sqrt(2 * math.pi)) * math.exp(-((math.log(D) - mu) ** 2) / (2 * sigma ** 2))


# Define the truncated log-normal PDF on [D_Min, D_Max]
def truncated_pdf(D, mu, sigma, F_DMin, F_DMax):
    return ln_PDF(D, mu, sigma) / (F_DMax - F_DMin)


# Compute the truncated CDF, integrating ln_PDF from D_Min to D and normalizing
def truncated_cdf(D, mu, sigma, D_Min, F_DMin, F_DMax):
    integral, _ = quad(lambda x: ln_PDF(x, mu, sigma), D_Min, D)
    return integral / (F_DMax - F_DMin)


def main():
    # --- Solicit user input with default values ---
    default_mu = math.log(2)  # Default mean of ln(D)
    default_sigma = 1.0  # Default sigma
    default_D_Min = 3.0 / 8.0  # Default small aperture
    default_D_Max = 1.0  # Default large aperture

    inp_mu = input(f"Enter mean ln(D) for pre-sieved rocks (default {default_mu:.3f}): ").strip()
    inp_sigma = input(f"Enter σ for pre-sieved rocks (default {default_sigma:.3f}): ").strip()
    inp_D_Min = input(f"Enter D_Min (small aperture, default {default_D_Min:.3f}): ").strip()
    inp_D_Max = input(f"Enter D_Max (large aperture, default {default_D_Max:.3f}): ").strip()

    mu = float(inp_mu) if inp_mu else default_mu
    sigma = float(inp_sigma) if inp_sigma else default_sigma
    D_Min = float(inp_D_Min) if inp_D_Min else default_D_Min
    D_Max = float(inp_D_Max) if inp_D_Max else default_D_Max

    # --- Compute normalization constants for the truncated distribution ---
    F_DMin, _ = quad(lambda D: ln_PDF(D, mu, sigma), 0, D_Min)
    F_DMax, _ = quad(lambda D: ln_PDF(D, mu, sigma), 0, D_Max)

    # Define the special D* value as specified (75% between D_Min and D_Max)
    D_star = D_Min + 0.75 * (D_Max - D_Min)
    F_D_star = truncated_cdf(D_star, mu, sigma, D_Min, F_DMin, F_DMax)

    # --- Create numpy arrays for D values ---
    D_vals = np.linspace(D_Min, D_Max, 500)
    pdf_vals = np.array([truncated_pdf(D, mu, sigma, F_DMin, F_DMax) for D in D_vals])
    cdf_vals = np.array([truncated_cdf(D, mu, sigma, D_Min, F_DMin, F_DMax) for D in D_vals])

    # --- Plotting ---
    plt.figure(figsize=(8, 10))

    # Upper Plot: Truncated Log-Normal PDF
    plt.subplot(2, 1, 1)
    plt.plot(D_vals, pdf_vals, 'b-', label='Truncated log-normal PDF')
    plt.xlim(D_Min, D_Max)
    plt.ylim(0, pdf_vals.max() * 1.1)
    plt.ylabel('f(D)')

    # Fill the area under the curve from D_Min to D_star
    D_fill = np.linspace(D_Min, D_star, 100)
    pdf_fill = np.array([truncated_pdf(D, mu, sigma, F_DMin, F_DMax) for D in D_fill])
    plt.fill_between(D_fill, pdf_fill, color='grey', alpha=0.3)

    # Add the PDF formula as an annotation using a single fraction
    text_x = D_Min + 0.1 * (D_Max - D_Min)
    text_y = pdf_vals.max() * 0.7
    plt.text(text_x, text_y,
             r'$f(D)=\frac{\frac{1}{D\sigma\sqrt{2\pi}}\exp\left[-\frac{(\ln D-\mu)^2}{2\sigma^2}\right]}{F_{D_{\max}}-F_{D_{\min}}}$',
             fontsize=10)

    # Annotate the filled area with F(D*) value
    arrow_x = D_star
    arrow_y = truncated_pdf(D_star, mu, sigma, F_DMin, F_DMax) * 0.5
    plt.annotate(r'$F(D^*)={:.3f}$'.format(F_D_star),
                 xy=(arrow_x, arrow_y),
                 xytext=(text_x, text_y * 0.5),
                 arrowprops=dict(arrowstyle='->'))

    plt.title('Truncated Log-Normal PDF and CDF')

    # Lower Plot: Truncated Log-Normal CDF
    plt.subplot(2, 1, 2)
    plt.plot(D_vals, cdf_vals, 'r-', label='Truncated log-normal CDF')
    plt.xlim(D_Min, D_Max)
    plt.ylim(0, 1)
    plt.xlabel('D')
    plt.ylabel(r'$F(D)=\frac{\int_{D_{\min}}^{D}\ln\_PDF(d)\,dd}{F_{D_{\max}}-F_{D_{\min}}}$')

    # Mark the point at D_star
    plt.plot(D_star, F_D_star, 'ko', markerfacecolor='white')
    plt.axvline(x=D_star, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=F_D_star, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()