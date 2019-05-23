{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}


module Main 
  ( main
  ) where


import           GHC.TypeNats (KnownNat)

import           Control.Monad (forM_, guard)

import           Options.Applicative
import           Data.Monoid ((<>))
import           Data.Maybe (mapMaybe)
import           Data.Ord (comparing)
import           Data.String (fromString)
import           Data.List (sortBy)
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy.IO as TL
import qualified Data.IORef as IORef

import qualified Numeric.LinearAlgebra.Static as LA

import qualified Dhall
-- import qualified DhallUtils as DU

import           System.FilePath (isAbsolute, (</>), (<.>))

import qualified Net.Graph as Graph
import qualified Net.Util as U
import qualified MWE
import qualified Format.Embedding as Emb
import qualified Format.Cupt as Cupt

import Debug.Trace (trace)


--------------------------------------------------
-- Commands
--------------------------------------------------


-- | Available commands
data Command
    = Train TrainConfig
      -- ^ Train a model
    | Tag TagConfig
      -- ^ Tagging
    | Clear Cupt.MweTyp
      -- ^ Tagging


-- | Training configuration
data TrainConfig = TrainConfig
  { trainCupt :: FilePath
  , trainEmbs :: FilePath
  -- , trainModelTyp :: MWE.Typ
  , trainMweCat :: T.Text
    -- ^ MWE category (e.g., LVC) to focus on
  , trainSgdCfgPath :: FilePath
    -- ^ SGD configuration path
--   , trainNetCfgPath :: FilePath
--     -- ^ Net configuration path
  , trainInModel   :: Maybe FilePath
    -- ^ Input model (otherwise, random)
  , trainOutModel  :: Maybe FilePath
    -- ^ Where to store the output model
  }


-- | Tagging configuration
data TagConfig = TagConfig
  { tagCupt :: FilePath
  , tagEmbs :: FilePath
  , tagMweCat :: T.Text
    -- ^ MWE category (e.g., LVC) to focus on
  , tagProb :: Double
    -- ^ MWE probability threshold
  , tagConstrained :: Bool
    -- ^ Use constrained global inference
  , tagModel   :: FilePath
    -- ^ Input model (otherwise, randomly initialized)
  }


--------------------------------------------------
-- Parse options
--------------------------------------------------


trainOptions :: Parser Command
trainOptions = fmap Train $ TrainConfig
  <$> strOption
        ( metavar "FILE"
       <> long "input"
       <> short 'i'
       <> help "Input .cupt file"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "embed"
       <> short 'e'
       <> help "Input embedding file"
        )
--   <*> option auto
--         ( long "typ"
--        <> help "MWE model type (Arc1, Arc2, ...)"
--         )
  <*> strOption
        ( long "mwe"
       <> short 't'
       <> help "MWE category (type) to learn"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "sgd"
       <> short 's'
       <> help "SGD configuration"
        )
--   <*> strOption
--         ( metavar "FILE"
--        <> long "net"
--        <> short 'n'
--        <> help "Network configuration"
--         )
  <*> (optional . strOption)
        ( metavar "FILE"
       <> long "model"
       <> short 'm'
       <> help "Input model"
        )
  <*> (optional . strOption)
        ( metavar "FILE"
       <> long "out-model"
       <> short 'o'
       <> help "Output model"
        )


tagOptions :: Parser Command
tagOptions = fmap Tag $ TagConfig
  <$> strOption
        ( metavar "FILE"
       <> long "input"
       <> short 'i'
       <> help "Input .cupt file"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "embed"
       <> short 'e'
       <> help "Input embedding file"
        )
  <*> strOption
        ( long "mwe"
       <> short 't'
       <> help "MWE category (type)"
        )
  <*> option auto
        ( long "prob"
       <> short 'p'
       <> value 0.5
       <> help "MWE probability threshold"
        )
  <*> flag False True
        ( long "constrained"
       <> short 'c'
       <> help "Use constrained global inference"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "model"
       <> short 'm'
       <> help "Input model"
        )


clearOptions :: Parser Command
clearOptions = Clear
  <$> strOption
        ( long "mwe"
       <> short 't'
       <> help "MWE category (type) to clear"
        )


--------------------------------------------------
-- Global options
--------------------------------------------------


opts :: Parser Command
opts = subparser
  ( command "train"
    (info (helper <*> trainOptions)
      (progDesc "Training")
    )
    <> command "tag"
    (info (helper <*> tagOptions)
      (progDesc "Tagging")
    )
    <> command "clear"
    (info (helper <*> clearOptions)
      (progDesc "Clear MWE annotations")
    )
  )


--------------------------------------------------
-- Main
--------------------------------------------------


-- | Run program depending on the cmdline arguments.
run :: Command -> IO ()
run cmd =
  case cmd of

    Train TrainConfig{..} -> do

      -- SGD configuration
      -- let (inputSettings, interpSettings) = DU.doubleSettings
      sgdCfg <-
        Dhall.input
        -- Dhall.inputWithSettings inputSettings
          Dhall.auto
          -- (Dhall.autoWith interpSettings)
          (dhallPath trainSgdCfgPath)
--         case trainSgdCfgPath of
--           Nothing -> return MWE.defTrainCfg
--           Just configPath ->
--             Dhall.input Dhall.auto (dhallPath configPath)
--       -- Network configuration
--       netCfg <- Dhall.input Dhall.auto (dhallPath trainNetCfgPath)

      -- Initial network
      net0 <-
        case trainInModel of
          Nothing -> do
            -- Extract the set of dependency labels
            depRelSet <- MWE.depRelsIn . concat
              <$> Cupt.readCupt trainCupt
            posTagSet <- MWE.posTagsIn . concat
              <$> Cupt.readCupt trainCupt
            MWE.new posTagSet depRelSet
          Just path -> U.load path
      -- Read .cupt (ignore paragraph boundaries)
      cupt <- map Cupt.decorate . concat
        <$> Cupt.readCupt trainCupt
      -- Read the corresponding embeddings
      embs <- Emb.readEmbeddings trainEmbs
      epochRef <- IORef.newIORef (0 :: Int)
      net <- MWE.train sgdCfg trainMweCat
        (mkInput cupt embs) net0 $ \net ->
          case trainOutModel of
            Nothing -> return ()
            Just pathBase -> do
              path <- do
                k <- IORef.readIORef epochRef
                IORef.writeIORef epochRef (k+1)
                return $ pathBase <.> show k
              U.save path net
              putStr "Saved the current model to: "
              putStrLn path
      case trainOutModel of
        Nothing -> return ()
        Just path -> do
          U.save path net
          putStr "Saved the final model to: "
          putStrLn path
      putStrLn "Done!"

    Tag TagConfig{..} -> do
      -- Load the model
      net <- U.load tagModel
      -- Read .cupt (ignore paragraph boundaries)
      cupt <- map Cupt.decorate . concat
        <$> Cupt.readCupt tagCupt
      -- Read the corresponding embeddings
      embs <- Emb.readEmbeddings tagEmbs
      MWE.tagManyIO cfg net
        (mkInput cupt embs)
      where
        cfg = MWE.TagConfig
          { MWE.mweTyp = tagMweCat
          , MWE.mweConstrained = tagConstrained
          }

    Clear mweTyp -> do
      cupt <- concat . Cupt.parseCupt <$> TL.getContents
      let cupt' = map (Cupt.removeMweAnnotations mweTyp) cupt
      forM_ cupt' $ \sent -> do
        T.putStr "# "
        T.putStrLn . T.unwords $ map Cupt.orth sent
        TL.putStrLn (Cupt.renderPar [sent])


main :: IO ()
main =
    execParser optsExt >>= run
  where
    optsExt = info (helper <*> opts)
       ( fullDesc
      <> progDesc "MWE identification tool"
      <> header "galago" )


--------------------------------------------------
-- Utils
--------------------------------------------------


mkInput
  :: (KnownNat d)
  => [Cupt.Sent]
  -> [Maybe [LA.R d]]
  -> [MWE.Sent d]
mkInput cupt embs =
  ( map
      (uncurry MWE.Sent)
      (mapMaybe clear $ zip cupt embs)
  ) 
  where
    clear (sent, mayEmbs) = report sent $ do
      embs <- mayEmbs
      -- guard $ simpleSent sent
      return (sent, embs)
    report sent mayVal = 
      case mayVal of
        Just val -> Just val
        Nothing -> 
          let sentTxt = T.intercalate " " (map Cupt.orth sent)
           in trace ("Ignoring sentence: " ++ T.unpack sentTxt) Nothing


dhallPath :: FilePath -> Dhall.Text
dhallPath path = fromString $
  if isAbsolute path
  then path
  else "./" </> path
