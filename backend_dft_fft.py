from bokeh.models.markers import X, Y
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pandas import DataFrame as df

import plotly


@st.cache
def _get_ecg_data():
    ecg_data_raw = np.loadtxt("ecg 100.dat", skiprows=2)
    ecg_time = ecg_data_raw[:, 0]
    ecg_norm = np.sum(ecg_data_raw[:, 1]) / len(ecg_data_raw[:, 1])
    # print(ecg_norm)
    ecg_data = ecg_data_raw[:, 1]
    fs = int(len(ecg_data) / (ecg_time[-1] - ecg_time[0]))
    print(fs)
    return ecg_time, ecg_data, fs


class Utils:
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
        X_power = np.abs(X) ** 2
        return {
            "Magnitude": np.array_split(X_magnitude_norm, 2)[0].flatten(),
            "Frequency": np.array_split(f, 2)[0].flatten(),
        }, {
            "Power": np.array_split(X_power, 2)[0],
            "Frequency": np.array_split(f, 2)[0],
        }

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
            "Windowed Waves": windowed_ecg[start : end + 1],
            "Window Name": window_name,
        }, {"Windowed Waves": windowed_ecg}

    def FFT_window(self, p, q, t, ecg, fs):
        p_fft, p_power = self.DFT_FFT_magnitude_norm(self.FFT(p["Windowed Waves"]), fs)
        q_fft, q_power = self.DFT_FFT_magnitude_norm(self.FFT(q["Windowed Waves"]), fs)
        t_fft, t_power = self.DFT_FFT_magnitude_norm(self.FFT(t["Windowed Waves"]), fs)
        ecg_fft, ecg_power = self.DFT_FFT_magnitude_norm(
            self.FFT(ecg["Windowed Waves"]), fs
        )
        return p_fft, q_fft, t_fft, ecg_fft, p_power, q_power, t_power, ecg_power

    def window_pack(self, two_a_state, fs, window_name, window_function, cl, cr):
        cl.markdown("#### " + window_name)
        container = cl.container()
        ecg, ecg_to_fft = self.ecg_window_plotter(
            "Single ECG Wave",
            two_a_state["P Start"],
            two_a_state["T End"],
            window_function,
            cl,
        )
        p, p_to_fft = self.ecg_window_plotter(
            "P Wave",
            two_a_state["P Start"],
            two_a_state["P End"],
            window_function,
            cl,
        )
        q, q_to_fft = self.ecg_window_plotter(
            "QRS Wave",
            two_a_state["QRS Start"],
            two_a_state["QRS End"],
            window_function,
            cl,
        )
        t, t_to_fft = self.ecg_window_plotter(
            "T Wave",
            two_a_state["T Start"],
            two_a_state["T End"],
            window_function,
            cl,
        )
        if two_a_state["Full Data"]:
            (
                p_fft,
                q_fft,
                t_fft,
                ecg_fft,
                p_power,
                q_power,
                t_power,
                ecg_power,
            ) = self.FFT_window(p_to_fft, q_to_fft, t_to_fft, ecg_to_fft, fs)
        else:
            (
                p_fft,
                q_fft,
                t_fft,
                ecg_fft,
                p_power,
                q_power,
                t_power,
                ecg_power,
            ) = self.FFT_window(p, q, t, ecg, fs)

        p_center, q_center, t_center, ecg_center = self.center_freq(
            p_fft, q_fft, t_fft, ecg_fft
        )
        self.center_freq_plotter(p_center, q_center, t_center, ecg_center, container)
        p_mean_power, q_mean_power, t_mean_power, ecg_mean_power = self.mean_power(
            p_power, q_power, t_power, ecg_power
        )

        mean_power_container = self.ecg_window_fft_plotter(
            p_fft, q_fft, t_fft, ecg_fft, window_name, cr
        )
        self.mean_power_plotter(
            p_mean_power,
            q_mean_power,
            t_mean_power,
            ecg_mean_power,
            mean_power_container,
        )

    def center_freq(self, p, q, t, ecg):
        p_center = np.sum(p["Frequency"] * p["Magnitude"]) / np.sum(p["Magnitude"])
        q_center = np.sum(q["Frequency"] * q["Magnitude"]) / np.sum(q["Magnitude"])
        t_center = np.sum(t["Frequency"] * t["Magnitude"]) / np.sum(t["Magnitude"])
        ecg_center = np.sum(ecg["Frequency"] * ecg["Magnitude"]) / np.sum(
            ecg["Magnitude"]
        )
        return p_center, q_center, t_center, ecg_center

    def mean_power(self, p_power, q_power, t_power, ecg_power):
        # fs = 360
        # f= 32768
        p_mean_power = np.sum(p_power["Power"] * p_power["Frequency"]) / np.sum(
            p_power["Power"]
        )
        q_mean_power = np.sum(q_power["Power"] * q_power["Frequency"]) / np.sum(
            q_power["Power"]
        )
        t_mean_power = np.sum(t_power["Power"] * t_power["Frequency"]) / np.sum(
            t_power["Power"]
        )
        ecg_mean_power = np.sum(ecg_power["Power"] * ecg_power["Frequency"]) / np.sum(
            ecg_power["Power"]
        )
        return p_mean_power, q_mean_power, t_mean_power, ecg_mean_power

    def center_freq_plotter(self, p_center, q_center, t_center, ecg_center, container):
        center_freq_df = df(
            [ecg_center, p_center, q_center, t_center],
            index=["ECG Wave", "P Wave", "QRS Wave", "T Wave"],
        )
        container.markdown("Center Frequencies")
        container.table(center_freq_df)

    def mean_power_plotter(
        self,
        p_mean_power,
        q_mean_power,
        t_mean_power,
        ecg_mean_power,
        mean_power_container,
    ):
        mean_power_df = df(
            [ecg_mean_power, p_mean_power, q_mean_power, t_mean_power],
            index=["ECG Wave", "P Wave", "QRS Wave", "T Wave"],
        )
        mean_power_container.markdown("Mean Power Frequency")
        mean_power_container.table(mean_power_df)

    def one_c_plotter(self, t, x, x_dft, x_fft):
        df_one_c = df({"Time": t, "Signal": x})
        df_fig = px.line(
            x=df_one_c["Time"],
            y=df_one_c["Signal"],
            color_discrete_sequence=["cyan"],
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
        dft_fig.update_traces(width=0.1)
        st.plotly_chart(dft_fig, use_container_width=True)

        fft_df = df({"Magnitude": x_fft["Magnitude"], "Frequency": x_fft["Frequency"]})
        fft_fig = px.bar(
            fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        fft_fig.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", title="FFT Magnitude"
        )
        fft_fig.update_traces(width=0.1)
        st.plotly_chart(fft_fig, use_container_width=True)

    def _ecg_plotter(self, raw_data_container):
        df_ecg = df({"Time": self.ecg_time, "Signal": self.ecg_data})
        df_fig = px.line(
            x=df_ecg["Time"],
            y=df_ecg["Signal"],
            color_discrete_sequence=["cyan"],
        )
        df_fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Amplitude",
            title="ECG Signal",
        )
        raw_data_container.plotly_chart(df_fig, use_container_width=True)

    def ecg_window_plotter(self, wave_name, start, end, window, c):
        windowed_ecg, windowed_ecg_to_fft = self.ecg_window(start, end, window)
        ecg_window_df = df(
            {
                "Time": self.ecg_time[start : end + 1],
                "Signal": windowed_ecg["Windowed Waves"],
            }
        )
        df_fig = px.line(
            ecg_window_df,
            x="Time",
            y="Signal",
            color_discrete_sequence=["cyan"],
        )
        df_fig.update_layout(
            xaxis_title="Time (t)",
            yaxis_title="Amplitude (mV)",
            title=wave_name + " " + windowed_ecg["Window Name"],
        )
        c.plotly_chart(df_fig, use_container_width=True)
        return windowed_ecg, windowed_ecg_to_fft

    def ecg_window_fft_plotter(self, p_fft, q_fft, t_fft, ecg_fft, window_name, c):
        p_fft_df = df(
            {"Frequency": p_fft["Frequency"], "Magnitude": p_fft["Magnitude"]}
        )
        p_fft_fig = px.line(
            p_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        p_fft_fig.update_layout(
            title="P Wave FFT (" + window_name + ")",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
        )

        q_fft_df = df(
            {"Frequency": q_fft["Frequency"], "Magnitude": q_fft["Magnitude"]}
        )
        q_fft_fig = px.line(
            q_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        q_fft_fig.update_layout(
            title="QRS Wave FFT (" + window_name + ")",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
        )

        t_fft_df = df(
            {"Frequency": t_fft["Frequency"], "Magnitude": t_fft["Magnitude"]}
        )
        t_fft_fig = px.line(
            t_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        t_fft_fig.update_layout(
            title="T Wave FFT (" + window_name + ")",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
        )

        ecg_fft_df = df(
            {
                "Frequency": ecg_fft["Frequency"],
                "Magnitude": ecg_fft["Magnitude"],
            }
        )
        ecg_fft_fig = px.line(
            ecg_fft_df, x="Frequency", y="Magnitude", color_discrete_sequence=["cyan"]
        )
        ecg_fft_fig.update_layout(
            title="ECG FFT (" + window_name + ")",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
        )
        c.markdown("#### " + window_name + " FFT")
        mean_power_container = c.container()
        c.plotly_chart(ecg_fft_fig, use_container_width=True)
        c.plotly_chart(p_fft_fig, use_container_width=True)
        c.plotly_chart(q_fft_fig, use_container_width=True)
        c.plotly_chart(t_fft_fig, use_container_width=True)
        return mean_power_container


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
        x_dft, _ = self.DFT_FFT_magnitude_norm(x_dft, fs)
        x_fft = self.FFT(x)
        x_fft, _ = self.DFT_FFT_magnitude_norm(x_fft, fs)

        self.one_c_plotter(t, x, x_dft, x_fft)

    def _run_two_a(self, two_a_state, raw_data_container):
        self.ecg_time, self.ecg_data, fs = _get_ecg_data()

        self._ecg_plotter(raw_data_container)
        cl, cr = st.columns(2)
        show_rec = cl.checkbox("Show Rectangular Window")
        show_han = cr.checkbox("Show Hanning Window")
        if show_rec:
            self.window_pack(
                two_a_state, fs, "Rectangular", self.rectangular_window, cl, cr
            )

        if show_han:
            self.window_pack(two_a_state, fs, "Hanning", self.hanning_window, cl, cr)
