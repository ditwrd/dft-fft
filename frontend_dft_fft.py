import streamlit as st
import numpy as np
import time
from backend_dft_fft import DFTFFTBackend


class Main(DFTFFTBackend):
    def __init__(self):
        self.main_page()

    def main_page(self):
        st.set_page_config(
            page_title="FFT Algorithm Exploration in ECG Data Frequency Domain Analyis",
            page_icon="📊",
            layout="wide",
        )
        st.title("FFT Algorithm Exploration in ECG Data Frequency Domain Analyis")

        self.one_a_frontend()
        self.one_b_frontend()
        self.one_c_frontend()
        self.one_d_frontend()
        self.two_a_frontend()

    def one_a_frontend(self):
        st.markdown(
            """
                     1. An analog signal x(t)=3.0+5.0*sin(2*pi*0.25*t) volt was sampled with 256 samples/second sampling rate.
                    - Realize computer program to sample the analog signal with the sampling rate.
                        """
        )
        with st.expander("1 a"):
            st.latex(r"x(t)=3.0+5.0sin{(2\pi 0.25t)}")
            st.latex(
                r"t=\frac{n}{fs}\rightarrow x(\frac{n}{fs})=3.0+5.0sin{(2\pi 0.25\frac{n}{fs})}"
            )
            st.latex(r"N=t\times fs")
            with st.form("one_a_form"):
                one_a_state = {}
                one_a_state["Sampling Frequency"] = st.number_input(
                    "Sampling Frequency", min_value=0, max_value=1024, value=256
                )
                one_a_state["Duration"] = st.number_input(
                    "Duration", min_value=0, max_value=100, value=1
                )
                one_a_state["Submitted"] = st.form_submit_button(
                    "Run", on_click=self._run_one_a(one_a_state)
                )

    def one_b_frontend(self):
        st.markdown(
            """ 
                    - Realize computer programs for frequency domain analysis using direct calculation (Discrete Fourier Transform), and Fast Fourier Transform Algorithm.
                    """
        )
        with st.expander("1 b"):
            st.markdown("### DFT")
            st.latex(r"X_{k}=\sum _{n=0}^{N-1}x_{n}e^{-{\frac {2\pi i}{N}}nk}")
            st.latex(r"X=Wx")
            st.latex(
                r"W=\left (\omega^{jk}  \right )_{j,k=0,...,N-1},\omega=e^{-2\pi i/N}"
            )
            st.latex(
                r"""W=\begin{bmatrix}
                1      & 1            & 1               & 1               & ...    & 1                   \\
                1      & \omega       & \omega^2        & \omega^3        & \cdots & \omega^{N-1}        \\
                1      & \omega^2     & \omega^4        & \omega^6        & \cdots & \omega^{2(N-1)}     \\
                1      & \omega^3     & \omega^6        & \omega^9        & \cdots & \omega^{3(N-1)}     \\
                \vdots & \vdots       & \vdots          & \vdots          & \ddots & \vdots              \\
                1      & \omega^{N-1} & \omega^{2(N-1)} & \omega^{3(N-1)} & \cdots & \omega^{(N-1)(N-1)}
                \end{bmatrix}"""
            )
            st.markdown("### FFT (Cooley-Tukey Algorithm (Radix 2))")
            st.latex(
                r"""{\begin{matrix}X_{k}&=&\sum \limits _{m=0}^{N/2-1}x_{2m}e^{-{\frac {2\pi i}{N}}(2m)k}+\sum \limits _{m=0}^{N/2-1}x_{2m+1}e^{-{\frac {2\pi i}{N}}(2m+1)k}\end{matrix}}"""
            )
            st.latex(
                r"""{\displaystyle {\begin{matrix}X_{k}=\underbrace {\sum \limits _{m=0}^{N/2-1}x_{2m}e^{-{\frac {2\pi i}{N/2}}mk}} _{\mathrm {DFT\;of\;even-indexed\;part\;of\;} x_{n}}{}+e^{-{\frac {2\pi i}{N}}k}\underbrace {\sum \limits _{m=0}^{N/2-1}x_{2m+1}e^{-{\frac {2\pi i}{N/2}}mk}} _{\mathrm {DFT\;of\;odd-indexed\;part\;of\;} x_{n}}=\underbrace{E_{k}+e^{-{\frac {2\pi i}{N}}k}O_{k}}_{for\;k=0,\cdots,\frac{N}{2}-1}\end{matrix}}}"""
            )
            st.latex(
                r"""{\displaystyle {\begin{aligned}X_{k+{\frac {N}{2}}}&=\sum \limits _{m=0}^{N/2-1}x_{2m}e^{-{\frac {2\pi i}{N/2}}m(k+{\frac {N}{2}})}+e^{-{\frac {2\pi i}{N}}(k+{\frac {N}{2}})}\sum \limits _{m=0}^{N/2-1}x_{2m+1}e^{-{\frac {2\pi i}{N/2}}m(k+{\frac {N}{2}})}\\&=\sum \limits _{m=0}^{N/2-1}x_{2m}e^{-{\frac {2\pi i}{N/2}}mk}e^{-2\pi mi}+e^{-{\frac {2\pi i}{N}}k}e^{-\pi i}\sum \limits _{m=0}^{N/2-1}x_{2m+1}e^{-{\frac {2\pi i}{N/2}}mk}e^{-2\pi mi}\\&=\sum \limits _{m=0}^{N/2-1}x_{2m}e^{-{\frac {2\pi i}{N/2}}mk}-e^{-{\frac {2\pi i}{N}}k}\sum \limits _{m=0}^{N/2-1}x_{2m+1}e^{-{\frac {2\pi i}{N/2}}mk}\\&=\underbrace{E_{k}-e^{-{\frac {2\pi i}{N}}k}O_{k}}_{for\;k=\frac{N}{2},\cdots,N-1}\end{aligned}}}"""
            )
            st.latex(
                r"""\begin{aligned}&X_k=E_{k}+e^{-{\frac {2\pi i}{N}}k}O_{k} \\ &X_{k+\frac{N}{2}}= E_{k}-e^{-{\frac {2\pi i}{N}}k}O_{k}\\ \end{aligned}"""
            )
            st.image("butterfly.png")
            st.markdown(
                """
                        ### Magnitude Normalization
                        - Magnitude of Fourier Transform is the square root of the sum of the squares of the real and imaginary parts.
                        - To normalize the magnitude we can use this method:
                            - DC Component (Frequency 0) normalize by 2
                            - All other components normalize by N/2
                        ### Program Realization
                        """
            )
            st.code(
                """
                    def padding(self, x):
                        log = np.log2(len(x))
                        return np.pad(
                            x, (0, int(2 ** ((log - log % 1) + 1) - len(x))), mode="constant"
                        ).flatten()

                    
                    def DFT(self, x):

                        if np.log2(len(x)) % 1 > 0:
                            x = self.padding(x)

                        N = len(x)
                        n = np.arange(N)
                        k = n[:, None]
                        W = np.exp(-2j * np.pi * n * k / N)
                        X = np.dot(W, x.reshape(-1, 1))
                        return X.flatten()

                    
                    def FFT(self, x):

                        if np.log2(len(x)) % 1 > 0:
                            x = self.padding(x)

                        x = np.asarray(x, dtype=float)
                        N = x.shape[0]

                        N_min = min(N, 2)

                        # DFT on all length-N_min sub-problems at once
                        n = np.arange(N_min)
                        k = n[:, None]
                        W = np.exp(-2j * np.pi * n * k / N_min)
                        X = np.dot(W, x.reshape((N_min, -1)))

                        # Recursive calculation all at once
                        while X.shape[0] < N:
                            X_even = X[:, : int(X.shape[1] / 2)]
                            X_odd = X[:, int(X.shape[1] / 2) :]
                            factor = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
                            factor.shape, factor
                            X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])
                        return X.flatten()

                    def DFT_FFT_magnitude_norm(self, X, fs):
                        N = len(X)
                        n = np.arange(N)
                        f = (n * fs / N).flatten()
                        X_norm = X
                        X_norm[1:] = X_norm[1:] * 2
                        X_norm = X_norm / N
                        X_magnitude_norm = np.abs(X_norm)
                        return {
                            "Magnitude": np.array_split(X_magnitude_norm, 2)[0].flatten(),
                            "Frequency": np.array_split(f, 2)[0].flatten(),
                        }
                    """
            )
            st.markdown(
                """
                        #### Sanity Check
                        Input = [0,1,2,..,2047]
                        """
            )
            c1, c2 = st.columns(2)
            c1.markdown(
                """
                        Numpy FFT == DFT
                        """
            )
            power_2 = 2 ** 10
            dft_start = time.time()
            dft_result = self.DFT(np.arange(power_2))
            dft_end = time.time()

            fft_start = time.time()
            fft_result = self.FFT(np.arange(power_2))
            fft_end = time.time()

            c1.write(np.allclose(np.fft.fft(np.arange(power_2)), dft_result))
            c1.text(f"{(dft_end-dft_start)*1000} ms")
            c2.markdown(
                """
                        Numpy FFT == FFT
                        """
            )
            c2.write(np.allclose(np.fft.fft(np.arange(power_2)), fft_result))
            c2.text(f"{(fft_end-fft_start)*1000} ms")

    def one_c_frontend(self):
        st.markdown(
            """- With sample length of 512 samples (duration of sampling: 2 seconds), perform DFT and FFT calculation to generate Frequency Domain data of your sequences. Identify contents of your signals in frequency domain data both in frequencies and magnitude. Check normalization of your DFT/FFT data can well represent the analog signal amplitude. Discuss your results, representation of period of signal is important in frequency domain analysis. """
        )
        with st.expander("1 c"):
            with st.form("one_c_form"):
                one_c_state = {}
                one_c_state["Sampling Frequency"] = st.number_input(
                    "Sampling Frequency", min_value=0, max_value=1024, value=256
                )
                one_c_state["Duration"] = st.number_input(
                    "Duration", min_value=0, max_value=100, value=2
                )
                one_c_state["Submitted"] = st.form_submit_button(
                    "Run", on_click=self._run_one_c(one_c_state)
                )

    def one_d_frontend(self):
        st.markdown(
            "- Do same task with sequences of 1024 and 2048 sample length. Write discussions about some important points about sample length and period of signal in case of  time to frequency domain transformation in discrete time data using DFT/FFT."
        )
        with st.expander("1 d"):
            with st.form("one_d_form_1"):
                one_d_state_1 = {}
                one_d_state_1["Sampling Frequency"] = st.number_input(
                    "Sampling Frequency", min_value=0, max_value=1024, value=256
                )
                one_d_state_1["Duration"] = st.number_input(
                    "Duration", min_value=0, max_value=100, value=4
                )
                one_d_state_1["Submitted"] = st.form_submit_button(
                    "Run", on_click=self._run_one_c(one_d_state_1)
                )

            with st.form("one_d_form_2"):
                one_d_state_2 = {}
                one_d_state_2["Sampling Frequency"] = st.number_input(
                    "Sampling Frequency", min_value=0, max_value=1024, value=256
                )
                one_d_state_2["Duration"] = st.number_input(
                    "Duration", min_value=0, max_value=100, value=8
                )
                one_d_state_2["Submitted"] = st.form_submit_button(
                    "Run", on_click=self._run_one_c(one_d_state_2)
                )

    def two_a_frontend(self):
        st.markdown(
            """
                    2. With ecg sequences ecg100.dat MLII, perform frequency domain analysis of P, QRS, and TU waves. Use rectangular and hanning window to localize the sequences of each wave.
                    - Compare frequency domain data of single cycle ECG, and each wave, identify the center frequency, and mean power frequency of each wave.
                    - Identify effect of rectangular and hanning window to frequency domain data.
                    """
        )
        with st.expander("2 a"):
            raw_data_container = st.container()

            two_a_state = {}
            c1, c2, c3 = st.columns(3)
            two_a_state["Preset Waves"] = c3.radio("Preset Waves", ["1", "2", "3"])
            two_a_state["Full Data"] = c3.checkbox("Full Data on FFT", False)
            fs = 360
            if two_a_state["Preset Waves"] == "1":
                preset = [
                    int(2.381 * fs),
                    int(2.514 * fs),
                    int(2.572 * fs),
                    int(2.656 * fs),
                    int(2.894 * fs),
                    int(3.131 * fs),
                ]
            elif two_a_state["Preset Waves"] == "2":
                preset = [
                    int(8.908 * fs),
                    int(9.014 * fs),
                    int(9.069 * fs),
                    int(9.153 * fs),
                    int(9.378 * fs),
                    int(9.578 * fs),
                ]
            elif two_a_state["Preset Waves"] == "3":
                preset = [
                    int(22.658 * fs),
                    int(22.794 * fs),
                    int(22.853 * fs),
                    int(22.931 * fs),
                    int(23.175 * fs),
                    int(23.4111 * fs),
                ]
            two_a_state["P Start"] = int(
                c1.number_input(
                    "P Start", min_value=0, max_value=21600, value=preset[0]
                )
            )
            two_a_state["P End"] = int(
                c2.number_input("P End", min_value=0, max_value=21600, value=preset[1])
            )
            two_a_state["QRS Start"] = int(
                c1.number_input(
                    "QRS Start", min_value=0, max_value=21600, value=preset[2]
                )
            )
            two_a_state["QRS End"] = int(
                c2.number_input(
                    "QRS End", min_value=0, max_value=21600, value=preset[3]
                )
            )
            two_a_state["T Start"] = int(
                c1.number_input(
                    "T Start", min_value=0, max_value=21600, value=preset[4]
                )
            )
            two_a_state["T End"] = int(
                c2.number_input("T End", min_value=0, max_value=21600, value=preset[5])
            )

            if two_a_state["P End"] - two_a_state["P Start"] < 0:
                st.error("P End must be greater than P Start")
            elif two_a_state["QRS End"] - two_a_state["QRS Start"] < 0:
                st.error("QRS End must be greater than QRS Start")
            elif two_a_state["T End"] - two_a_state["T Start"] < 0:
                st.error("T End must be greater than T Start")
            else:
                self._run_two_a(two_a_state, raw_data_container)


if __name__ == "__main__":
    main = Main()
