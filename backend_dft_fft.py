import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pandas import DataFrame as df
# from icecream import ic
import plotly


@st.cache
def _get_ecg_data():
    ecg_data_raw = np.loadtxt("ecg 100.dat", skiprows=2)
    ecg_time = ecg_data_raw[:, 0]
    ecg_norm = np.sum(ecg_data_raw[:, 1]) / len(ecg_data_raw[:, 1])
    # ecg_norm = 0
    ecg_data = ecg_data_raw[:, 1] + np.abs(ecg_norm)
    fs=len(ecg_data) / (ecg_time[-1] - ecg_time[0])
    return ecg_time, ecg_data,fs


class Utils:
    def padding(self, x):
        log = np.log2(len(x))
        return np.pad(
            x, (0, int(2 ** ((log - log % 1) + 1) - len(x))), mode="constant"
        ).flatten()

    # @st.cache
    def DFT(self, x):

        if np.log2(len(x)) % 1 > 0:
            x = self.padding(x)

        N = len(x)
        n = np.arange(N)
        k = n[:, None]
        W = np.exp(-2j * np.pi * n * k / N)
        X = np.dot(W, x.reshape(-1, 1))
        return X.flatten()

    # @st.cache
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

        # build-up each level of the recursive calculation all at once
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
        X_magnitude = np.abs(X)

        X_magnitude_norm = np.append(X_magnitude[0] / 2, X_magnitude[1:] / (N / 2))
        # ic(X_magnitude_norm)
        return {"Magnitude": X_magnitude_norm.flatten(), "Frequency": f}

    @st.cache(allow_output_mutation=True)
    def get_sine_wave(self, state):
        fs = state["Sampling Frequency"]
        t = state["Duration"]
        N = t * fs
        t = np.linspace(0, t, int(N))
        x = 3.0 + 5.0 * np.sin(2 * np.pi * 0.25 * t)
        return fs, t, x

    @st.cache(allow_output_mutation=True)
    def rectangular_window(self, N):
        return np.ones(N), "Rectangular"

    @st.cache(allow_output_mutation=True)
    def hanning_window(self, N):
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N))), "Hanning"

    @st.cache(allow_output_mutation=True)
    def window_padding(self, window, start, end):
        return np.pad(window, (start, len(self.ecg_data) - end - 1), "constant")

    @st.cache(allow_output_mutation=True)
    def ecg_window(self, start, end, window_func):
        N = end - start + 1
        window, window_name = window_func(N)
        window = self.window_padding(window, start, end)
        windowed_ecg = self.ecg_data * window
        return {
            "Windowed Waves": windowed_ecg,
            "Window Name": window_name,
        }

    @st.cache
    def FFT_rec(self, p_rec, q_rec, t_rec, rec_ecg,fs):
        p_rec_fft = self.DFT_FFT_magnitude_norm(self.FFT(p_rec["Windowed Waves"]),fs)
        q_rec_fft = self.DFT_FFT_magnitude_norm(self.FFT(q_rec["Windowed Waves"]),fs)
        t_rec_fft = self.DFT_FFT_magnitude_norm(self.FFT(t_rec["Windowed Waves"]),fs)
        rec_ecg_fft = self.DFT_FFT_magnitude_norm(self.FFT(rec_ecg),fs)
        return p_rec_fft, q_rec_fft, t_rec_fft, rec_ecg_fft

    @st.cache
    def FFT_han(self, p_han, q_han, t_han, han_ecg,fs):
        p_han_fft = self.DFT_FFT_magnitude_norm(self.FFT(p_han["Windowed Waves"]),fs)
        q_han_fft = self.DFT_FFT_magnitude_norm(self.FFT(q_han["Windowed Waves"]),fs)
        t_han_fft = self.DFT_FFT_magnitude_norm(self.FFT(t_han["Windowed Waves"]),fs)
        han_ecg_fft = self.DFT_FFT_magnitude_norm(self.FFT(han_ecg),fs)
        return p_han_fft, q_han_fft, t_han_fft, han_ecg_fft

    def one_c_plotter(self, t, x, x_dft, x_fft):
        df_one_c = df({"Time": t, "Signal": x})
        df_fig = px.line(
            x=df_one_c["Time"], y=df_one_c["Signal"], color_discrete_sequence=["cyan"]
        )
        df_fig.update_layout(
            xaxis_title="Time", yaxis_title="Amplitude", title="Signal"
        )
        st.plotly_chart(df_fig, use_container_width=True)

        dft_df = df({"Magnitude": x_dft["Magnitude"], "Frequency": x_dft["Frequency"]})
        dft_fig = px.bar(
            dft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        dft_fig.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", title="DFT Magnitude"
        )
        st.plotly_chart(dft_fig, use_container_width=True)

        fft_df = df({"Magnitude": x_fft["Magnitude"], "Frequency": x_fft["Frequency"]})
        fft_fig = px.bar(
            fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        fft_fig.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", title="FFT Magnitude"
        )
        st.plotly_chart(fft_fig, use_container_width=True)

    def _ecg_plotter(self, raw_data_container):
        df_ecg = df({"Time": self.ecg_time, "Signal": self.ecg_data})
        df_fig = px.line(
            x=df_ecg["Time"], y=df_ecg["Signal"], color_discrete_sequence=["cyan"]
        )
        df_fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Amplitude",
            title="ECG Signal",
        )
        raw_data_container.plotly_chart(df_fig, use_container_width=True)

    def ecg_window_plotter(self, wave_name, start, end, window):
        windowed_ecg = self.ecg_window(start, end, window)
        ecg_window_df = df(
            {"Time": self.ecg_time, "Signal": windowed_ecg["Windowed Waves"]}
        )
        df_fig = px.line(
            ecg_window_df, x="Time", y="Signal", color_discrete_sequence=["cyan"]
        )
        df_fig.update_layout(
            xaxis_title="Time (t)",
            yaxis_title="Amplitude (mV)",
            title=wave_name + " " + windowed_ecg["Window Name"],
        )
        st.plotly_chart(df_fig, use_container_width=True)
        return windowed_ecg

    def rec_plotter(self, p_rec_fft, q_rec_fft, t_rec_fft, rec_ecg_fft):
        p_rec_fft_df = df(
            {"Frequency": p_rec_fft["Frequency"], "Magnitude": p_rec_fft["Magnitude"]}
        )
        p_rec_fft_fig = px.bar(
            p_rec_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        q_rec_fft_df = df(
            {"Frequency": q_rec_fft["Frequency"], "Magnitude": q_rec_fft["Magnitude"]}
        )
        q_rec_fft_fig = px.bar(
            q_rec_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        t_rec_fft_df = df(
            {"Frequency": t_rec_fft["Frequency"], "Magnitude": t_rec_fft["Magnitude"]}
        )
        t_rec_fft_fig = px.bar(
            t_rec_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        rec_ecg_fft_df = df(
            {
                "Frequency": rec_ecg_fft["Frequency"],
                "Magnitude": rec_ecg_fft["Magnitude"],
            }
        )
        rec_ecg_fft_fig = px.bar(
            rec_ecg_fft_df,
            x="Frequency",
            y="Magnitude",
            color_discrete_sequence=["cyan"],
        )

        st.plotly_chart(p_rec_fft_fig, use_container_width=True)
        st.plotly_chart(q_rec_fft_fig, use_container_width=True)
        st.plotly_chart(t_rec_fft_fig, use_container_width=True)
        st.plotly_chart(rec_ecg_fft_fig, use_container_width=True)

    def han_plotter(self, p_han_fft, q_han_fft, t_han_fft, han_ecg_fft):
        p_han_fft_df = df(
            {"Frequency": p_han_fft["Frequency"], "Magnitude": p_han_fft["Magnitude"]}
        )
        p_han_fft_fig = px.bar(
            p_han_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        q_han_fft_df = df(
            {"Frequency": q_han_fft["Frequency"], "Magnitude": q_han_fft["Magnitude"]}
        )
        q_han_fft_fig = px.bar(
            q_han_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        t_han_fft_df = df(
            {"Frequency": t_han_fft["Frequency"], "Magnitude": t_han_fft["Magnitude"]}
        )
        t_han_fft_fig = px.bar(
            t_han_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        han_ecg_fft_df = df(
            {
                "Frequency": han_ecg_fft["Frequency"],
                "Magnitude": han_ecg_fft["Magnitude"],
            }
        )
        han_ecg_fft_fig = px.bar(
            han_ecg_fft_df,
            x="Frequency",
            y="Magnitude",
            color_discrete_sequence=["cyan"],
        )

        st.plotly_chart(p_han_fft_fig, use_container_width=True)
        st.plotly_chart(q_han_fft_fig, use_container_width=True)
        st.plotly_chart(t_han_fft_fig, use_container_width=True)
        st.plotly_chart(han_ecg_fft_fig, use_container_width=True)


class DFTFFTBackend(Utils):
    def _run_one_a(self, one_a_state):
        fs, t, x = self.get_sine_wave(one_a_state)
        df_one_a = df({"Time": t, "Signal": x})
        df_fig = px.line(
            x=df_one_a["Time"], y=df_one_a["Signal"], color_discrete_sequence=["cyan"]
        )
        df_fig.update_layout(xaxis_title="Time", yaxis_title="Amplitude")
        st.plotly_chart(df_fig, use_container_width=True)

    def _run_one_c(self, one_c_state):
        fs, t, x = self.get_sine_wave(one_c_state)
        x_dft = self.DFT(x)
        x_dft = self.DFT_FFT_magnitude_norm(x_dft, fs)
        x_fft = self.FFT(x)
        x_fft = self.DFT_FFT_magnitude_norm(x_fft, fs)

        self.one_c_plotter(t, x, x_dft, x_fft)

    def _run_two_a(self, two_a_state, raw_data_container):
        self.ecg_time, self.ecg_data,fs = _get_ecg_data()

        self._ecg_plotter(raw_data_container)
        c_rec, c_han = st.columns(2)
        show_rec = c_rec.checkbox("Show Rectangular Window")
        show_han = c_han.checkbox("Show Hanning Window")
        if show_rec:
            rec_container = st.empty()

            p_rec = self.ecg_window_plotter(
                "P Wave",
                two_a_state["P Start"],
                two_a_state["P End"],
                self.rectangular_window,
            )
            q_rec = self.ecg_window_plotter(
                "QRS Wave",
                two_a_state["QRS Start"],
                two_a_state["QRS End"],
                self.rectangular_window,
            )
            t_rec = self.ecg_window_plotter(
                "T Wave",
                two_a_state["T Start"],
                two_a_state["T End"],
                self.rectangular_window,
            )
            rec_ecg = (
                p_rec["Windowed Waves"]
                + q_rec["Windowed Waves"]
                + t_rec["Windowed Waves"]
            )

            df_rec = df({"Time": self.ecg_time, "Signal": rec_ecg})
            df_fig_rec = px.line(
                df_rec, x="Time", y="Signal", color_discrete_sequence=["cyan"]
            )
            rec_container.markdown("#### Rectangular")
            rec_container.plotly_chart(df_fig_rec, use_container_width=True)
            
            with st.spinner("Calculating Rectangular FFT..."):
                p_rec_fft, q_rec_fft, t_rec_fft, rec_ecg_fft = self.FFT_rec(
                    p_rec, q_rec, t_rec, rec_ecg,fs
                )

            self.rec_plotter(p_rec_fft, q_rec_fft, t_rec_fft, rec_ecg_fft)
        if show_han:
            han_container = st.empty()

            p_han = self.ecg_window_plotter(
                "P Wave",
                two_a_state["P Start"],
                two_a_state["P End"],
                self.hanning_window,
            )
            q_han = self.ecg_window_plotter(
                "QRS Wave",
                two_a_state["QRS Start"],
                two_a_state["QRS End"],
                self.hanning_window,
            )
            t_han = self.ecg_window_plotter(
                "T Wave",
                two_a_state["T Start"],
                two_a_state["T End"],
                self.hanning_window,
            )
            han_ecg = (
                p_han["Windowed Waves"]
                + q_han["Windowed Waves"]
                + t_han["Windowed Waves"]
            )

            df_han = df({"Signal": han_ecg, "Time": self.ecg_time})
            df_fig_han = px.line(
                df_han, x="Time", y="Signal", color_discrete_sequence=["cyan"]
            )
            han_container.markdown("#### Hanning")
            han_container.plotly_chart(df_fig_han, use_container_width=True)
            
            with st.spinner("Calculating Hanning FFT..."):
                p_han_fft, q_han_fft, t_han_fft, han_ecg_fft = self.FFT_han(
                    p_han, q_han, t_han, rec_ecg,fs
                )
            self.han_plotter(p_han_fft, q_han_fft, t_han_fft, han_ecg_fft)
